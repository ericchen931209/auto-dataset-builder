"""
Neural DQS — Feature Extraction

Computes f(D) = [AQ, SH, LD, PD, CB] ∈ ℝ⁵ for a dataset directory.
SH (Sharpness) replaces DS (Diversity) — Laplacian variance is a reliable
image-quality indicator that correctly degrades with blur/noise.
See docs/dqs-model.md for full mathematical formulation.
"""

import os
import logging
import math
from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DQSFeatures:
    annotation_quality: float   # AQ ∈ [0,1]  annotation completeness + geometry
    sharpness: float            # IQ ∈ [0,1]  image quality (blur × noise-clean)
    clip_diversity: float       # CD ∈ [0,1]  CLIP-based semantic diversity
    lighting_diversity: float   # LD ∈ [0,1]
    pose_diversity: float       # PD ∈ [0,1]
    class_balance: float        # CB ∈ [0,1]

    def to_vector(self) -> list[float]:
        return [
            self.annotation_quality,
            self.sharpness,
            self.clip_diversity,
            self.lighting_diversity,
            self.pose_diversity,
            self.class_balance,
        ]

    def to_dict(self) -> dict:
        return {
            "annotation_quality": self.annotation_quality,
            "sharpness": self.sharpness,
            "clip_diversity": self.clip_diversity,
            "lighting_diversity": self.lighting_diversity,
            "pose_diversity": self.pose_diversity,
            "class_balance": self.class_balance,
        }


# ─── Individual feature functions ────────────────────────────────────────────

def compute_annotation_quality(labels_dir: str) -> float:
    """
    AQ: composite of annotation completeness and geometric consistency.

    completeness = fraction of label files that have ≥1 valid annotation
    geometry     = fraction of bboxes with reasonable area (0.5%–70%)

    AQ = 0.6 * completeness + 0.4 * geometry
    This correctly penalises datasets where many images have no labels at all.
    """
    label_files = list(Path(labels_dir).glob("*.txt"))
    if not label_files:
        return 0.0

    files_with_annots = 0
    reasonable = 0
    total_boxes = 0

    for lf in label_files:
        lines = [l for l in lf.read_text().strip().splitlines() if l.strip()]
        valid = 0
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            w, h = float(parts[3]), float(parts[4])
            area = w * h
            total_boxes += 1
            if 0.005 <= area <= 0.70:
                reasonable += 1
                valid += 1
        if valid > 0:
            files_with_annots += 1

    completeness = files_with_annots / len(label_files)
    geometry = (reasonable / total_boxes) if total_boxes > 0 else 0.0
    return 0.6 * completeness + 0.4 * geometry


def compute_lighting_diversity(image_dir: str, n_buckets: int = 3) -> float:
    """
    LD: Normalized entropy of brightness distribution.
    Buckets: dark (<85), normal (85-170), bright (>170).
    """
    image_paths = list(Path(image_dir).glob("*.jpg")) + list(Path(image_dir).glob("*.png"))
    if not image_paths:
        return 0.0

    buckets = [0, 0, 0]

    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mean_v = hsv[:, :, 2].mean()

        if mean_v < 85:
            buckets[0] += 1
        elif mean_v < 170:
            buckets[1] += 1
        else:
            buckets[2] += 1

    total = sum(buckets)
    if total == 0:
        return 0.0

    entropy = 0.0
    for count in buckets:
        if count > 0:
            p = count / total
            entropy -= p * math.log(p)

    return entropy / math.log(n_buckets)  # normalized to [0,1]


def compute_pose_diversity(labels_dir: str, n_buckets: int = 3) -> float:
    """
    PD: Normalized entropy of bbox aspect ratio distribution.
    Captures viewpoint variation (front/side/overhead).
    """
    label_files = list(Path(labels_dir).glob("*.txt"))
    if not label_files:
        return 0.0

    # buckets: tall (r<0.5), square (0.5<=r<=2), wide (r>2)
    buckets = [0, 0, 0]

    for lf in label_files:
        for line in lf.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            w, h = float(parts[3]), float(parts[4])
            if h == 0:
                continue
            r = w / h
            if r < 0.5:
                buckets[0] += 1
            elif r <= 2.0:
                buckets[1] += 1
            else:
                buckets[2] += 1

    total = sum(buckets)
    if total == 0:
        return 0.0

    entropy = 0.0
    for count in buckets:
        if count > 0:
            p = count / total
            entropy -= p * math.log(p)

    return entropy / math.log(n_buckets)


def compute_class_balance(labels_dir: str) -> float:
    """
    CB: 1 - normalized Gini coefficient of class distribution.
    = 1 when perfectly balanced, approaches 0 for single-class.
    For single-class datasets this is always 1.0.
    """
    label_files = list(Path(labels_dir).glob("*.txt"))
    if not label_files:
        return 0.0

    class_counts: dict[int, int] = {}
    for lf in label_files:
        for line in lf.read_text().strip().splitlines():
            parts = line.split()
            if not parts:
                continue
            cls = int(parts[0])
            class_counts[cls] = class_counts.get(cls, 0) + 1

    if not class_counts:
        return 0.0

    n_classes = len(class_counts)
    if n_classes == 1:
        return 1.0  # single-class dataset: perfectly "balanced" by definition

    total = sum(class_counts.values())
    proportions = [c / total for c in class_counts.values()]
    gini = 1 - sum(p ** 2 for p in proportions)
    max_gini = 1 - 1 / n_classes  # gini for uniform distribution

    return gini / max_gini if max_gini > 0 else 1.0


def compute_diversity_clip(image_dir: str, sample_size: int = 100) -> float:
    """
    DS: CLIP-based diversity score.
    Requires open_clip to be installed; falls back to pixel-based estimate.
    """
    try:
        import torch
        from PIL import Image as PILImage

        model, preprocess = _get_clip_model()

        image_paths = list(Path(image_dir).glob("*.jpg")) + list(Path(image_dir).glob("*.png"))
        if len(image_paths) < 2:
            return 0.0

        # Sample for efficiency
        if len(image_paths) > sample_size:
            indices = np.random.choice(len(image_paths), sample_size, replace=False)
            image_paths = [image_paths[i] for i in indices]

        embeddings = []
        for path in image_paths:
            try:
                img = preprocess(PILImage.open(path).convert("RGB")).unsqueeze(0)
                with torch.no_grad():
                    emb = model.encode_image(img)
                    emb = emb / emb.norm(dim=-1, keepdim=True)
                embeddings.append(emb.squeeze().numpy())
            except Exception:
                continue

        if len(embeddings) < 2:
            return 0.0

        emb_matrix = np.stack(embeddings)
        # Mean pairwise cosine distance (1 - similarity) = diversity
        sim_matrix = emb_matrix @ emb_matrix.T
        n = len(embeddings)
        off_diag = sim_matrix[np.triu_indices(n, k=1)]
        diversity = float(1.0 - off_diag.mean())
        return max(0.0, min(1.0, diversity))

    except ImportError:
        logger.warning("open_clip not installed — using pixel-based diversity fallback")
        return _pixel_diversity_fallback(image_dir, sample_size)


def _pixel_diversity_fallback(image_dir: str, sample_size: int = 50) -> float:
    """Fallback: mean pairwise L2 distance of downsampled grayscale histograms."""
    image_paths = (
        list(Path(image_dir).glob("*.jpg")) + list(Path(image_dir).glob("*.png"))
    )[:sample_size]

    hists = []
    for path in image_paths:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        hist = cv2.calcHist([img], [0], None, [32], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-8)
        hists.append(hist)

    if len(hists) < 2:
        return 0.0

    distances = []
    for i in range(len(hists)):
        for j in range(i + 1, len(hists)):
            dist = float(cv2.compareHist(
                hists[i].astype(np.float32),
                hists[j].astype(np.float32),
                cv2.HISTCMP_BHATTACHARYYA,
            ))
            distances.append(dist)

    return min(1.0, float(np.mean(distances)))


def compute_sharpness(image_dir: str, sample_size: int = 100) -> float:
    """
    Image Quality (IQ) = geometric mean of blur_score and noise_cleanliness.

    blur_score:       Laplacian variance after median denoise, capped at 500.
                      Low for blurry images, high for sharp ones.
    noise_cleanliness: 1 - normalised high-frequency residual std.
                      Low for noisy images, high for clean ones.

    IQ = sqrt(blur_score × noise_cleanliness) ∈ [0,1].
    This way, EITHER blur OR noise degrades the score, independently.
    """
    BLUR_CAP = 500.0
    NOISE_SIGMA_CAP = 25.0   # residual std above this → score ≈ 0

    image_paths = (
        list(Path(image_dir).glob("*.jpg")) + list(Path(image_dir).glob("*.png"))
    )[:sample_size]

    scores = []
    for path in image_paths:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Blur score
        denoised = cv2.medianBlur(img, 3)
        blur_score = min(float(cv2.Laplacian(denoised, cv2.CV_64F).var()) / BLUR_CAP, 1.0)

        # Noise cleanliness
        smooth = cv2.GaussianBlur(img, (5, 5), 1.5)
        noise_std = float(np.std(img.astype(np.float32) - smooth.astype(np.float32)))
        noise_cleanliness = max(0.0, 1.0 - noise_std / NOISE_SIGMA_CAP)

        iq = math.sqrt(blur_score * noise_cleanliness)
        scores.append(iq)

    return float(np.mean(scores)) if scores else 0.0


# ─── CLIP model cache (loaded once per process) ───────────────────────────────

_clip_cache: dict = {}


def _get_clip_model():
    """Load and cache CLIP model (ViT-B-32) once per process."""
    if "model" not in _clip_cache:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        model.eval()
        _clip_cache["model"] = model
        _clip_cache["preprocess"] = preprocess
    return _clip_cache["model"], _clip_cache["preprocess"]


# ─── Main extractor ───────────────────────────────────────────────────────────

def extract_features(image_dir: str, labels_dir: str) -> DQSFeatures:
    """
    Compute all six DQS features for a dataset.
    image_dir: directory containing .jpg/.png images
    labels_dir: directory containing YOLO .txt label files
    """
    aq  = compute_annotation_quality(labels_dir)
    sh  = compute_sharpness(image_dir)
    cd  = compute_diversity_clip(image_dir)
    ld  = compute_lighting_diversity(image_dir)
    pd_ = compute_pose_diversity(labels_dir)
    cb  = compute_class_balance(labels_dir)

    features = DQSFeatures(
        annotation_quality=aq,
        sharpness=sh,
        clip_diversity=cd,
        lighting_diversity=ld,
        pose_diversity=pd_,
        class_balance=cb,
    )

    logger.info(
        f"DQS features — AQ={aq:.3f} IQ={sh:.3f} CD={cd:.3f} "
        f"LD={ld:.3f} PD={pd_:.3f} CB={cb:.3f}"
    )
    return features
