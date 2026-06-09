import os
import logging
from pathlib import Path
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Default thresholds
BLUR_THRESHOLD = 100.0      # Laplacian variance; below = blurry
DARK_THRESHOLD = 20.0       # Mean pixel value (0-255); below = too dark
BRIGHT_THRESHOLD = 0.98     # Fraction of pixels > 250; above = overexposed
MIN_DIMENSION = 64          # Minimum width or height in pixels


@dataclass
class ImageQualityReport:
    path: str
    blur_score: float
    brightness_mean: float
    overexposed_ratio: float
    width: int
    height: int
    is_blurry: bool
    is_dark: bool
    is_overexposed: bool
    is_too_small: bool

    @property
    def is_clean(self) -> bool:
        return not (self.is_blurry or self.is_dark or self.is_overexposed or self.is_too_small)

    @property
    def rejection_reason(self) -> str | None:
        reasons = []
        if self.is_blurry:
            reasons.append(f"blurry(score={self.blur_score:.1f})")
        if self.is_dark:
            reasons.append(f"dark(mean={self.brightness_mean:.1f})")
        if self.is_overexposed:
            reasons.append(f"overexposed(ratio={self.overexposed_ratio:.2f})")
        if self.is_too_small:
            reasons.append(f"too_small({self.width}x{self.height})")
        return ", ".join(reasons) if reasons else None


def analyze_image(
    image_path: str,
    blur_threshold: float = BLUR_THRESHOLD,
    dark_threshold: float = DARK_THRESHOLD,
    bright_threshold: float = BRIGHT_THRESHOLD,
    min_dimension: int = MIN_DIMENSION,
) -> ImageQualityReport:
    """Compute quality metrics for a single image."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return ImageQualityReport(
            path=image_path, blur_score=0, brightness_mean=0,
            overexposed_ratio=0, width=0, height=0,
            is_blurry=True, is_dark=True, is_overexposed=False, is_too_small=True
        )

    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Blur: Laplacian variance
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Brightness: mean of V channel in HSV
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    brightness_mean = float(hsv[:, :, 2].mean())

    # Overexposure: fraction of pixels with V > 250
    overexposed_ratio = float((hsv[:, :, 2] > 250).mean())

    return ImageQualityReport(
        path=image_path,
        blur_score=blur_score,
        brightness_mean=brightness_mean,
        overexposed_ratio=overexposed_ratio,
        width=w,
        height=h,
        is_blurry=blur_score < blur_threshold,
        is_dark=brightness_mean < dark_threshold,
        is_overexposed=overexposed_ratio > bright_threshold,
        is_too_small=(w < min_dimension or h < min_dimension),
    )


@dataclass
class CleaningReport:
    total: int
    kept: int
    removed: int
    removed_blurry: int = 0
    removed_dark: int = 0
    removed_overexposed: int = 0
    removed_too_small: int = 0
    reports: list[ImageQualityReport] = field(default_factory=list)


def clean_dataset(
    image_dir: str,
    labels_dir: str | None = None,
    blur_threshold: float = BLUR_THRESHOLD,
    dark_threshold: float = DARK_THRESHOLD,
    bright_threshold: float = BRIGHT_THRESHOLD,
    min_dimension: int = MIN_DIMENSION,
    dry_run: bool = False,
) -> CleaningReport:
    """
    Analyze all images in image_dir and remove low-quality ones.
    If labels_dir is provided, also deletes the corresponding .txt label file.
    dry_run=True: report only, don't delete.
    """
    image_paths = sorted(
        p for p in Path(image_dir).iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )

    reports: list[ImageQualityReport] = []
    removed_counts = {"blurry": 0, "dark": 0, "overexposed": 0, "too_small": 0}

    for path in image_paths:
        report = analyze_image(
            str(path), blur_threshold, dark_threshold, bright_threshold, min_dimension
        )
        reports.append(report)

        if not report.is_clean:
            logger.debug(f"Removing {path.name}: {report.rejection_reason}")

            if not dry_run:
                os.remove(path)
                # Remove matching label file if it exists
                if labels_dir:
                    label_path = Path(labels_dir) / (path.stem + ".txt")
                    if label_path.exists():
                        os.remove(label_path)

            if report.is_blurry:
                removed_counts["blurry"] += 1
            if report.is_dark:
                removed_counts["dark"] += 1
            if report.is_overexposed:
                removed_counts["overexposed"] += 1
            if report.is_too_small:
                removed_counts["too_small"] += 1

    removed = sum(1 for r in reports if not r.is_clean)

    return CleaningReport(
        total=len(reports),
        kept=len(reports) - removed,
        removed=removed,
        removed_blurry=removed_counts["blurry"],
        removed_dark=removed_counts["dark"],
        removed_overexposed=removed_counts["overexposed"],
        removed_too_small=removed_counts["too_small"],
        reports=reports,
    )
