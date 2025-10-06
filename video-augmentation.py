import concurrent.futures
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
import pyrallis
from tqdm import tqdm


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

        self.augmentation_factories = {
            "rotate": self._build_rotate_augmenter,
            "brightness": self._build_brightness_augmenter,
            "dropout": self._build_coarse_dropout_augmenter,
            "motion_blur": self._build_motion_blur_augmenter,
            "perspective": self._build_perspective_augmenter,
            "noise": self._build_noise_augmenter,
        }

    def iter_video_aug_tasks(self):
        """(video_path, rel_path, video_name, video_suffix, aug_method, aug_idx, output_path)
        for each video and augmentation method."""
        supported_formats = [".mp4", ".avi", ".mov", ".mkv"]
        video_count = 0
        for file_path in self.input_path.glob("**/*"):
            if file_path.is_file() and file_path.suffix.lower() in supported_formats:
                rel_path = file_path.relative_to(self.input_path)
                video_name = file_path.stem
                video_suffix = file_path.suffix
                for aug_idx, aug_method in enumerate(
                    self.augmentation_factories.keys()
                ):
                    aug_dir = self.output_path / aug_method
                    if self.preserve_structure:
                        output_dir = aug_dir / rel_path.parent
                    else:
                        output_dir = aug_dir
                    os.makedirs(output_dir, exist_ok=True)
                    output_filename = (
                        f"{video_name}_{aug_method}_{aug_idx}{video_suffix}"
                    )
                    output_path = output_dir / output_filename
                    yield (
                        file_path,
                        rel_path,
                        video_name,
                        video_suffix,
                        aug_method,
                        aug_idx,
                        output_path,
                    )
                video_count += 1
        print(f"Found {video_count} videos to process")
        return

    def _build_rotate_augmenter(self, rng):
        angle = float(rng.uniform(-12.0, 12.0))
        return iaa.Affine(rotate=angle), f"angle={angle:.1f}°"

    def _build_brightness_augmenter(self, rng):
        factor = float(rng.uniform(0.85, 1.15))
        return iaa.MultiplyBrightness(factor), f"factor={factor:.2f}"

    def _build_coarse_dropout_augmenter(self, rng):
        drop_prob = float(rng.uniform(0.03, 0.08))
        region_size = float(rng.uniform(0.10, 0.18))
        info = f"p={drop_prob:.2f}, size={region_size:.2f}"
        return iaa.CoarseDropout(p=drop_prob, size_percent=region_size), info

    def _build_motion_blur_augmenter(self, rng):
        kernel = int(rng.choice([5, 7, 9]))
        angle = float(rng.uniform(-20.0, 20.0))
        return iaa.MotionBlur(k=kernel, angle=angle), f"k={kernel}, angle={angle:.1f}°"

    def _build_perspective_augmenter(self, rng):
        scale = float(rng.uniform(0.04, 0.10))
        shear = float(rng.uniform(-8.0, 8.0))
        info = f"scale={scale:.3f}, shear={shear:.1f}°"
        return (
            iaa.Sequential(
                [iaa.PerspectiveTransform(scale=scale), iaa.Affine(shear=shear)]
            ),
            info,
        )

    def _build_noise_augmenter(self, rng):
        sigma = float(rng.uniform(8.0, 20.0))
        return iaa.AdditiveGaussianNoise(scale=sigma), f"sigma={sigma:.1f}"

    def _process_video_batch(self, frames, augmenter):
        frames_array = np.array(frames)
        augmented_frames = augmenter.augment_images(frames_array)
        return list(augmented_frames)

    def _augment_video(
        self,
        video_path,
        rel_path,
        video_name,
        video_suffix,
        aug_method,
        aug_idx,
        output_path,
    ):
        try:
            seed_value = 42 + hash(str(video_path) + aug_method) % 10000
            ia.seed(seed_value)

            rng = np.random.default_rng(seed_value)
            factory = self.augmentation_factories[aug_method]
            base_augmenter, magnitude_info = factory(rng)
            print(f"{video_name} | {aug_method}: {magnitude_info}")
            augmenter = base_augmenter.to_deterministic()

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"Error: Could not open video {video_path}")
                return False, None

            fps_val = cap.get(cv2.CAP_PROP_FPS)
            fps = int(fps_val) if fps_val > 0 else 30

            w_val = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            width = int(w_val) if w_val > 0 else None

            h_val = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            height = int(h_val) if h_val > 0 else None

            cf = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            total_frames = int(cf) if cf > 0 else 0

            inferred_frame = None
            if width is None or height is None:
                ret_tmp, frame_tmp = cap.read()
                if ret_tmp and frame_tmp is not None:
                    inferred_frame = frame_tmp
                    height, width = frame_tmp.shape[:2]
                else:
                    width, height = 640, 480
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(
                str(output_path),
                fourcc,
                fps,
                (width, height),
            )

            frames_batch = []
            frame_count = 0
            skipped_frames = 0
            last_good_frame = None

            tqdm_total = total_frames if total_frames > 0 else None
            with tqdm(
                total=tqdm_total,
                desc=f"Augmenting {video_name} with {aug_method}",
            ) as pbar:
                while cap.isOpened():
                    try:
                        ret, frame = cap.read()
                    except Exception:
                        ret, frame = False, None

                    if not ret or frame is None:
                        skipped_frames += 1
                        if last_good_frame is not None:
                            substitute = last_good_frame.copy()
                        elif inferred_frame is not None:
                            substitute = inferred_frame.copy()
                        else:
                            substitute = np.zeros((height, width, 3), dtype=np.uint8)

                        try:
                            frame_rgb = cv2.cvtColor(substitute, cv2.COLOR_BGR2RGB)
                        except Exception:
                            if (
                                isinstance(substitute, np.ndarray)
                                and substitute.ndim == 3
                                and substitute.shape[2] == 3
                            ):
                                frame_rgb = substitute
                            else:
                                frame_rgb = np.zeros((height, width, 3), dtype=np.uint8)

                        frames_batch.append(frame_rgb)
                        frame_count += 1
                    else:
                        try:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        except Exception:
                            skipped_frames += 1
                            if last_good_frame is not None:
                                frame_rgb = last_good_frame.copy()
                            elif inferred_frame is not None:
                                frame_rgb = cv2.cvtColor(
                                    inferred_frame, cv2.COLOR_BGR2RGB
                                )
                            else:
                                frame_rgb = np.zeros((height, width, 3), dtype=np.uint8)

                        frames_batch.append(frame_rgb)
                        last_good_frame = frame_rgb
                        frame_count += 1

                    if len(frames_batch) == self.batch_size or (
                        total_frames > 0 and frame_count == total_frames
                    ):
                        try:
                            augmented_frames = self._process_video_batch(
                                frames_batch, augmenter
                            )
                        except Exception as e:
                            print(
                                f"Augmentation failed for a batch in {video_path}: {e}."
                                " Writing original frames."
                            )
                            augmented_frames = frames_batch

                        written = 0
                        for aug_frame in augmented_frames:
                            try:
                                aug_frame_bgr = cv2.cvtColor(
                                    aug_frame, cv2.COLOR_RGB2BGR
                                )
                                out.write(aug_frame_bgr)
                                written += 1
                            except Exception:
                                skipped_frames += 1

                        frames_batch = []
                        try:
                            pbar.update(written)
                        except Exception:
                            pass

            if skipped_frames > 0:
                print(
                    f"Note: skipped or substituted {skipped_frames} frames in {video_path}"
                )

            cap.release()
            out.release()

            return True, output_path

        except Exception as e:
            print(f"Error processing {video_path} with {aug_method}: {e}")
            return False, None

    def run(self):
        tasks = list(self.iter_video_aug_tasks())
        if not tasks:
            print("No videos found in the input directory")
            return []

        print(
            f"Applying all {len(self.augmentation_factories)} augmentation methods to each video"
        )

        augmented_paths = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(self._augment_video, *task) for task in tasks]
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
