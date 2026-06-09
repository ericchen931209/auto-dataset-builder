import os
import logging
from pathlib import Path
from dataclasses import dataclass, field

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    video_path: str
    output_dir: str
    total_frames: int
    extracted_frames: int
    frame_paths: list[str] = field(default_factory=list)


def extract_fixed_rate(
    video_path: str,
    output_dir: str,
    fps: float = 1.0,
) -> ExtractionResult:
    """Extract frames at a fixed rate (e.g. 1 frame per second)."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, int(video_fps / fps))

    frame_idx = 0
    saved_paths: list[str] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            out_path = os.path.join(output_dir, f"frame_{frame_idx:07d}.jpg")
            cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_paths.append(out_path)
        frame_idx += 1

    cap.release()
    logger.info(f"Fixed-rate: extracted {len(saved_paths)}/{total_frames} frames from {video_path}")

    return ExtractionResult(
        video_path=video_path,
        output_dir=output_dir,
        total_frames=total_frames,
        extracted_frames=len(saved_paths),
        frame_paths=saved_paths,
    )


def extract_adaptive(
    video_path: str,
    output_dir: str,
    ssim_threshold: float = 0.85,
    min_interval_sec: float = 0.5,
) -> ExtractionResult:
    """
    Extract frames only when the scene changes significantly.
    Compares consecutive frames using SSIM; saves when SSIM < ssim_threshold.

    ssim_threshold: lower = extract more frames (0.0 = always, 1.0 = never)
    min_interval_sec: minimum gap between extractions to avoid burst duplicates
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    min_frame_gap = int(video_fps * min_interval_sec)

    prev_gray: np.ndarray | None = None
    frame_idx = 0
    last_saved_idx = -min_frame_gap
    saved_paths: list[str] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.resize(gray, (320, 180))

        should_save = False
        if prev_gray is None:
            should_save = True
        elif (frame_idx - last_saved_idx) >= min_frame_gap:
            score, _ = ssim(prev_gray, gray_small, full=True)
            if score < ssim_threshold:
                should_save = True

        if should_save:
            out_path = os.path.join(output_dir, f"frame_{frame_idx:07d}.jpg")
            cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_paths.append(out_path)
            prev_gray = gray_small
            last_saved_idx = frame_idx

        frame_idx += 1

    cap.release()
    logger.info(f"Adaptive: extracted {len(saved_paths)}/{total_frames} frames from {video_path}")

    return ExtractionResult(
        video_path=video_path,
        output_dir=output_dir,
        total_frames=total_frames,
        extracted_frames=len(saved_paths),
        frame_paths=saved_paths,
    )
