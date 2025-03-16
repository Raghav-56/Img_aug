import cv2
import os
import numpy as np
import time
import shutil
import random
import concurrent.futures
from tqdm import tqdm
from pathlib import Path
import albumentations as A
import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F
from dataclasses import dataclass, field
import pyrallis
from typing import Optional, List
import sys


@dataclass
class AugmentationConfig:
    """Configuration for video augmentation"""

    input: str = field(
        default_factory=lambda: None,
        metadata={"help": "Input directory containing videos"},
    )
    """Input directory containing videos"""

    output: str = field(
        default_factory=lambda: None,
        metadata={"help": "Output directory for augmented videos"},
    )
    """Output directory for augmented videos"""

    batch_size: int = 32
    """Number of frames to process at once"""

    preserve_structure: bool = True
    """Preserve directory structure in output"""


class VideoAugmenter:
    """A class to handle video dataset augmentation with multiple techniques using modern libraries"""

    def __init__(
        self,
        input_path="DIC/Samvedna_Sample/Sample_vid",
        output_path="DIC/Samvedna_Sample/Sample_vid_aug",
        batch_size=32,
        preserve_structure=True,
    ):
        """
        Initialize the VideoAugmenter

        Args:
            input_path: Path to the input videos folder
            output_path: Path to save augmented videos
            batch_size: Number of frames to process at once for better performance
            preserve_structure: Whether to preserve the directory structure
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.batch_size = batch_size
        self.video_paths = []
        self.preserve_structure = preserve_structure

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create output directory if it doesn't exist
        os.makedirs(self.output_path, exist_ok=True)

        # Define all augmentation strategies
        self.augmentation_methods = {
            "crop_pad": self._get_crop_pad_augmenter,
            "elastic": self._get_elastic_augmenter,
            "scale": self._get_scale_augmenter,
            "affine": self._get_piecewise_affine_augmenter,
            "translate": self._get_translate_augmenter,
            "hflip": self._get_horizontal_flip_augmenter,
            "vflip": self._get_vertical_flip_augmenter,
            "rotate": self._get_rotate_augmenter,
            "perspective": self._get_perspective_augmenter,
            "brightness": self._get_brightness_augmenter,
            "contrast": self._get_contrast_augmenter,
            "dropout": self._get_coarse_dropout_augmenter,
            "edge": self._get_edge_detection_augmenter,
            "motion_blur": self._get_motion_blur_augmenter,
            "sharpen": self._get_sharpen_augmenter,
            "emboss": self._get_emboss_augmenter,
            "gaussian_blur": self._get_gaussian_blur_augmenter,
            "hue_saturation": self._get_hue_saturation_augmenter,
            "invert": self._get_invert_augmenter,
            "noise": self._get_noise_augmenter,
        }

    def _find_videos(self):
        """Find all videos in the input directory and preserve structure"""
        supported_formats = [".mp4", ".avi", ".mov", ".mkv"]
        self.video_paths = []

        for file_path in self.input_path.glob("**/*"):
            if file_path.is_file() and file_path.suffix.lower() in supported_formats:
                # Save both absolute path and relative path to preserve structure
                rel_path = file_path.relative_to(self.input_path)
                self.video_paths.append((file_path, rel_path))

        print(f"Found {len(self.video_paths)} videos to process")
        return len(self.video_paths) > 0

    # Modern augmentation method definitions using a mix of torchvision and albumentations
    def _get_crop_pad_augmenter(self):
        """Crop and pad augmentation"""
        return A.Compose(
            [A.CropAndPad(percent=(-0.1, 0.1), pad_mode_cv=random.choice([0, 1, 2]))]
        )

    def _get_elastic_augmenter(self):
        """Elastic transformation augmentation"""
        return A.Compose([A.ElasticTransform(alpha=1.0, sigma=50.0, p=1.0)])

    def _get_scale_augmenter(self):
        """Scale augmentation"""
        return A.Compose([A.RandomScale(scale_limit=(0.8, 1.2), p=1.0)])

    def _get_piecewise_affine_augmenter(self):
        """Piecewise affine augmentation using GridDistortion as alternative"""
        return A.Compose([A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0)])

    def _get_translate_augmenter(self):
        """Translation augmentation"""
        return A.Compose(
            [A.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, p=1.0)]
        )

    def _get_horizontal_flip_augmenter(self):
        """Horizontal flip augmentation"""
        return A.Compose([A.HorizontalFlip(p=1.0)])

    def _get_vertical_flip_augmenter(self):
        """Vertical flip augmentation"""
        return A.Compose([A.VerticalFlip(p=1.0)])

    def _get_rotate_augmenter(self):
        """Rotation augmentation"""
        return A.Compose([A.Rotate(limit=30, p=1.0)])

    def _get_perspective_augmenter(self):
        """Perspective transformation and shear augmentation"""
        return A.Compose(
            [
                A.Perspective(scale=(0.05, 0.1), p=1.0),
                A.Affine(shear=(-15, 15), p=1.0),
            ]
        )

    def _get_brightness_augmenter(self):
        """Brightness change augmentation"""
        return A.Compose(
            [
                A.RandomBrightnessContrast(
                    brightness_limit=(0.2, 0.4), contrast_limit=0, p=1.0
                )
            ]
        )

    def _get_contrast_augmenter(self):
        """Contrast change augmentation"""
        return A.Compose(
            [
                A.RandomBrightnessContrast(
                    brightness_limit=0, contrast_limit=(0.5, 1.5), p=1.0
                )
            ]
        )

    def _get_coarse_dropout_augmenter(self):
        """
        Coarse dropout augmentation similar to imgaug implementation.
        Based on the imgaug.augmenters.arithmetic.CoarseDropout parameters.
        """
        return A.Compose([A.GridDropout(size_px=(2, 16))])

    def _get_edge_detection_augmenter(self):
        """Edge detection augmentation"""

        def apply_edge_detection(img):
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(img_gray, 100, 200)
            edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            return cv2.addWeighted(img, 0.7, edges_rgb, 0.3, 0)

        return apply_edge_detection

    def _get_motion_blur_augmenter(self):
        """Motion blur augmentation"""
        return A.Compose([A.MotionBlur(blur_limit=(7, 15), p=1.0)])

    def _get_sharpen_augmenter(self):
        """Sharpen augmentation"""
        return A.Compose([A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0)])

    def _get_emboss_augmenter(self):
        """Emboss augmentation - implemented manually since not directly in Albumentations"""

        def apply_emboss(img):
            kernel = np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]])
            emboss = cv2.filter2D(img, -1, kernel)
            return cv2.addWeighted(img, 0.7, emboss, 0.3, 0)

        return apply_emboss

    def _get_gaussian_blur_augmenter(self):
        """Gaussian blur augmentation"""
        return A.Compose([A.GaussianBlur(blur_limit=(3, 7), p=1.0)])

    def _get_hue_saturation_augmenter(self):
        """Hue and saturation change augmentation"""
        return A.Compose(
            [
                A.HueSaturationValue(
                    hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0
                )
            ]
        )

    def _get_invert_augmenter(self):
        """Invert augmentation"""
        return A.Compose([A.InvertImg(p=1.0)])

    def _get_noise_augmenter(self):
        """Add noise augmentation"""
        return A.Compose([A.GaussNoise(var_limit_=(10.0, 50.0), p=1.0)])

    def _process_frame_with_albumentations(self, frame, augmenter):
        """Process a frame with Albumentations augmenter"""
        if isinstance(augmenter, A.Compose):
            augmented = augmenter(image=frame)
            return augmented["image"]
        else:
            # For custom function augmenters like emboss and edge detection
            return augmenter(frame)

    def _process_video_batch(self, frames, augmenter):
        """Process a batch of frames with the given augmenter"""
        augmented_frames = []
        for frame in frames:
            augmented_frames.append(
                self._process_frame_with_albumentations(frame, augmenter)
            )
        return augmented_frames

    def _augment_video(self, video_info, aug_method, aug_idx):
        """
        Augment a single video using the specified augmentation method

        Args:
            video_info: Tuple of (absolute_path, relative_path) to the input video
            aug_method: Augmentation method name
            aug_idx: Index for the output filename

        Returns:
            Tuple of (success, output_path)
        """
        video_path, rel_path = video_info
        try:
            # Get augmenter
            augmenter = self.augmentation_methods[aug_method]()

            # Prepare file paths
            video_name = video_path.stem
            output_filename = f"{video_name}_{aug_method}_{aug_idx}{video_path.suffix}"

            # Create output directory structure if preserving structure
            if self.preserve_structure:
                output_dir = self.output_path / rel_path.parent
                os.makedirs(output_dir, exist_ok=True)
                output_path = output_dir / output_filename
            else:
                output_path = self.output_path / output_filename

            # Open video
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"Error: Could not open video {video_path}")
                return False, None

            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

            # Process video in batches
            frames_batch = []
            frame_count = 0

            with tqdm(
                total=total_frames, desc=f"Augmenting {video_name} with {aug_method}"
            ) as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Convert BGR to RGB for compatibility with albumentations
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames_batch.append(frame)
                    frame_count += 1

                    # Process batch when it reaches the specified size
                    if (
                        len(frames_batch) == self.batch_size
                        or frame_count == total_frames
                    ):
                        augmented_frames = self._process_video_batch(
                            frames_batch, augmenter
                        )
                        for aug_frame in augmented_frames:
                            # Convert RGB back to BGR for OpenCV
                            aug_frame_bgr = cv2.cvtColor(aug_frame, cv2.COLOR_RGB2BGR)
                            out.write(aug_frame_bgr)

                        frames_batch = []
                        pbar.update(len(augmented_frames))

            # Release resources
            cap.release()
            out.release()

            return True, output_path

        except Exception as e:
            print(f"Error processing {video_path} with {aug_method}: {e}")
            return False, None

    def run(self):
        """
        Run the augmentation process

        Returns:
            List of paths to augmented videos
        """
        if not self._find_videos():
            print("No videos found in the input directory")
            return []

        augmented_paths = []

        # Apply all available augmentations
        augmentation_list = list(self.augmentation_methods.keys())
        print(
            f"Applying all {len(augmentation_list)} augmentation methods to each video"
        )

        # Process videos with concurrent execution using ProcessPoolExecutor for better CPU parallelism
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []

            for video_info in self.video_paths:
                for aug_idx, aug_method in enumerate(augmentation_list):
                    futures.append(
                        executor.submit(
                            self._augment_video, video_info, aug_method, aug_idx
                        )
                    )

            # Collect results
            for future in concurrent.futures.as_completed(futures):
                success, output_path = future.result()
                if success and output_path:
                    augmented_paths.append(output_path)

        return augmented_paths


def main():
    """Main function to parse arguments and run the augmenter"""
    try:
        config = pyrallis.parse(config_class=AugmentationConfig)

        # Check if required arguments are provided
        if config.input is None or config.output is None:
            print("Error: Required arguments missing.")
            print(
                "Usage: python video-augmentation.py --input INPUT_DIR --output OUTPUT_DIR [--batch_size BATCH_SIZE] [--preserve_structure PRESERVE_STRUCTURE]"
            )
            print(
                "Example: python video-augmentation.py --input ./videos --output ./augmented_videos"
            )
            sys.exit(1)

        start_time = time.time()

        # Initialize and run the augmenter
        augmenter = VideoAugmenter(
            config.input,
            config.output,
            config.batch_size,
            config.preserve_structure,
        )

        augmented_paths = augmenter.run()

        # Print summary
        elapsed_time = time.time() - start_time
        print(f"Augmentation completed in {elapsed_time:.2f} seconds")
        print(f"Created {len(augmented_paths)} augmented videos")

    except pyrallis.utils.ParsingError as e:
        print("Error parsing arguments:")
        print(
            "Usage: python video-augmentation.py --input INPUT_DIR --output OUTPUT_DIR [--batch_size BATCH_SIZE] [--preserve_structure PRESERVE_STRUCTURE]"
        )
        print(
            "Example: python video-augmentation.py --input ./videos --output ./augmented_videos"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
