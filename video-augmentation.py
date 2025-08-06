import cv2
import os
import numpy as np
import time
import concurrent.futures
from tqdm import tqdm
from pathlib import Path
import imgaug as ia
import imgaug.augmenters as iaa
from dataclasses import dataclass, field
import pyrallis
import sys


@dataclass
class AugmentationConfig:
    input: str = field(
        default_factory=lambda: None,
        metadata={"help": "Input directory containing videos"},
    )
    output: str = field(
        default_factory=lambda: None,
        metadata={"help": "Output directory for augmented videos"},
    )
    batch_size: int = 32
    preserve_structure: bool = True


class VideoAugmenter:
    def __init__(
        self,
        input_path="DIC/Samvedna_Sample/Sample_vid",
        output_path="DIC/Samvedna_Sample/Sample_vid_aug",
        batch_size=32,
        preserve_structure=True,
    ):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.batch_size = batch_size
        self.video_paths = []
        self.preserve_structure = preserve_structure

        os.makedirs(self.output_path, exist_ok=True)

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
        supported_formats = [".mp4", ".avi", ".mov", ".mkv"]
        self.video_paths = []

        for file_path in self.input_path.glob("**/*"):
            if file_path.is_file() and file_path.suffix.lower() in supported_formats:
                rel_path = file_path.relative_to(self.input_path)
                self.video_paths.append((file_path, rel_path))

        print(f"Found {len(self.video_paths)} videos to process")
        return len(self.video_paths) > 0

    def _get_crop_pad_augmenter(self):
        # Fixed parameters - no tuples or ranges that would create randomness
        return iaa.CropAndPad(percent=-0.2, pad_mode="constant")

    def _get_elastic_augmenter(self):
        return iaa.ElasticTransformation(alpha=2.5, sigma=50.0)

    def _get_scale_augmenter(self):
        return iaa.Affine(scale=0.75)

    def _get_piecewise_affine_augmenter(self):
        return iaa.PiecewiseAffine(scale=0.07)

    def _get_translate_augmenter(self):
        # Fixed translation values
        return iaa.Affine(translate_percent={"x": 0.2, "y": 0.2})

    def _get_horizontal_flip_augmenter(self):
        return iaa.Fliplr(1.0)

    def _get_vertical_flip_augmenter(self):
        return iaa.Flipud(1.0)

    def _get_rotate_augmenter(self):
        return iaa.Affine(rotate=30)

    def _get_perspective_augmenter(self):
        return iaa.Sequential(
            [iaa.PerspectiveTransform(scale=0.15), iaa.Affine(shear=20)]
        )

    def _get_brightness_augmenter(self):
        return iaa.MultiplyBrightness(1.5)

    def _get_contrast_augmenter(self):
        return iaa.LinearContrast(1.75)

    def _get_coarse_dropout_augmenter(self):
        # Fixed dropout parameters
        return iaa.CoarseDropout(
            p=0.1,  # percentage of pixels to drop
            size_percent=0.2,  # size of the dropped areas
        )

    def _get_edge_detection_augmenter(self):
        return iaa.EdgeDetect(alpha=1.0)

    def _get_motion_blur_augmenter(self):
        return iaa.MotionBlur(k=15, angle=45)  # Fixed angle

    def _get_sharpen_augmenter(self):
        return iaa.Sharpen(alpha=0.8, lightness=1.0)

    def _get_emboss_augmenter(self):
        return iaa.Emboss(alpha=1.0, strength=1.5)

    def _get_gaussian_blur_augmenter(self):
        return iaa.GaussianBlur(sigma=3.0)

    def _get_hue_saturation_augmenter(self):
        return iaa.AddToHueAndSaturation(30)

    def _get_invert_augmenter(self):
        return iaa.Invert(1.0)

    def _get_noise_augmenter(self):
        return iaa.AdditiveGaussianNoise(scale=50)

    def _process_video_batch(self, frames, augmenter):
        frames_array = np.array(frames)
        augmented_frames = augmenter.augment_images(frames_array)
        return list(augmented_frames)

    def _augment_video(self, video_info, aug_method, aug_idx):
        video_path, rel_path = video_info
        try:
            # Set seed for reproducibility
            ia.seed(42 + hash(str(video_path) + aug_method) % 10000)

            # Create the augmenter with fixed parameters
            base_augmenter = self.augmentation_methods[aug_method]()

            # Make it deterministic - this ensures the SAME transformation
            # is applied to every frame
            augmenter = base_augmenter.to_deterministic()

            video_name = video_path.stem
            output_filename = f"{video_name}_{aug_method}_{aug_idx}{video_path.suffix}"

            if self.preserve_structure:
                output_dir = self.output_path / rel_path.parent
                os.makedirs(output_dir, exist_ok=True)
                output_path = output_dir / output_filename
            else:
                output_path = self.output_path / output_filename

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"Error: Could not open video {video_path}")
                return False, None

            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

            frames_batch = []
            frame_count = 0

            with tqdm(
                total=total_frames, desc=f"Augmenting {video_name} with {aug_method}"
            ) as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames_batch.append(frame)
                    frame_count += 1

                    if (
                        len(frames_batch) == self.batch_size
                        or frame_count == total_frames
                    ):
                        augmented_frames = self._process_video_batch(
                            frames_batch, augmenter
                        )
                        for aug_frame in augmented_frames:
                            aug_frame_bgr = cv2.cvtColor(aug_frame, cv2.COLOR_RGB2BGR)
                            out.write(aug_frame_bgr)

                        frames_batch = []
                        pbar.update(len(augmented_frames))

            cap.release()
            out.release()

            return True, output_path

        except Exception as e:
            print(f"Error processing {video_path} with {aug_method}: {e}")
            return False, None

    def run(self):
        if not self._find_videos():
            print("No videos found in the input directory")
            return []

        augmented_paths = []
        augmentation_list = list(self.augmentation_methods.keys())
        print(
            f"Applying all {len(augmentation_list)} augmentation methods to each video"
        )

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []

            for video_info in self.video_paths:
                for aug_idx, aug_method in enumerate(augmentation_list):
                    futures.append(
                        executor.submit(
                            self._augment_video, video_info, aug_method, aug_idx
                        )
                    )

            for future in concurrent.futures.as_completed(futures):
                success, output_path = future.result()
                if success and output_path:
                    augmented_paths.append(output_path)

        return augmented_paths


def main():
    try:
        config = pyrallis.parse(config_class=AugmentationConfig)

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

        augmenter = VideoAugmenter(
            config.input,
            config.output,
            config.batch_size,
            config.preserve_structure,
        )

        augmented_paths = augmenter.run()

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
