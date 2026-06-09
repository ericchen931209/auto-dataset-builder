"""
Uncertainty Sampler — Active Learning Stage.

Scans a YOLO labels directory and identifies images where the annotation
confidence is low, indicating the model is uncertain.  These images are
candidates for re-annotation in the next active learning iteration.

Uncertainty measures supported:
  "min_conf"  — image score = minimum box confidence  (most common)
  "mean_conf" — image score = mean box confidence
  "entropy"   — image score = -sum(p * log(p)) over per-class counts
                (measures class distribution uncertainty)
"""
import math
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ImageUncertainty:
    image_path: str
    label_path: str
    score: float          # lower = more uncertain
    num_boxes: int
    min_confidence: float
    mean_confidence: float


def _parse_yolo_label(label_path: str) -> list[dict]:
    """Parse a YOLO .txt label file into list of box dicts."""
    boxes = []
    try:
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                box: dict = {
                    "class_id": int(parts[0]),
                    "cx": float(parts[1]),
                    "cy": float(parts[2]),
                    "w":  float(parts[3]),
                    "h":  float(parts[4]),
                    # confidence is optional (6th column added by yolo_annotator)
                    "conf": float(parts[5]) if len(parts) > 5 else 1.0,
                }
                boxes.append(box)
    except (OSError, ValueError) as e:
        logger.warning(f"Could not parse {label_path}: {e}")
    return boxes


def _entropy_score(boxes: list[dict]) -> float:
    """Shannon entropy of per-class box counts (normalised to [0,1])."""
    if not boxes:
        return 0.0
    from collections import Counter
    counts = Counter(b["class_id"] for b in boxes)
    total = sum(counts.values())
    probs = [c / total for c in counts.values()]
    raw = -sum(p * math.log2(p) for p in probs if p > 0)
    max_entropy = math.log2(max(len(counts), 2))
    return raw / max_entropy if max_entropy > 0 else 0.0


def sample_uncertain_images(
    images_dir: str,
    labels_dir: str,
    strategy: str = "min_conf",
    top_k: int | None = None,
    threshold: float = 0.5,
) -> list[ImageUncertainty]:
    """
    Identify the most uncertain images in a dataset split.

    An image is considered uncertain when its uncertainty score is below
    `threshold`.  If `top_k` is set, return at most the `top_k` most
    uncertain images (sorted ascending by score).

    Args:
        images_dir: Directory containing image files (.jpg/.png/.jpeg).
        labels_dir: Directory containing YOLO .txt label files.
        strategy: "min_conf" | "mean_conf" | "entropy"
        top_k: Maximum number of images to return (None = all uncertain).
        threshold: Score cutoff; images with score < threshold are uncertain.
                   For "entropy" strategy a *higher* score means more uncertain,
                   so the comparison is inverted automatically.

    Returns:
        List of ImageUncertainty sorted by score ascending
        (most uncertain first, or descending for entropy).
    """
    images_dir_path = Path(images_dir)
    labels_dir_path = Path(labels_dir)

    image_files = sorted(
        p for p in images_dir_path.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )

    uncertain: list[ImageUncertainty] = []

    for img_path in image_files:
        lbl_path = labels_dir_path / (img_path.stem + ".txt")
        if not lbl_path.exists():
            # No label → treat as fully uncertain
            uncertain.append(ImageUncertainty(
                image_path=str(img_path),
                label_path=str(lbl_path),
                score=0.0,
                num_boxes=0,
                min_confidence=0.0,
                mean_confidence=0.0,
            ))
            continue

        boxes = _parse_yolo_label(str(lbl_path))
        if not boxes:
            uncertain.append(ImageUncertainty(
                image_path=str(img_path),
                label_path=str(lbl_path),
                score=0.0,
                num_boxes=0,
                min_confidence=0.0,
                mean_confidence=0.0,
            ))
            continue

        confs = [b["conf"] for b in boxes]
        min_conf  = min(confs)
        mean_conf = sum(confs) / len(confs)

        if strategy == "min_conf":
            score = min_conf
            is_uncertain = score < threshold
        elif strategy == "mean_conf":
            score = mean_conf
            is_uncertain = score < threshold
        elif strategy == "entropy":
            score = _entropy_score(boxes)
            is_uncertain = score > threshold   # high entropy = uncertain
        else:
            raise ValueError(f"Unknown strategy: {strategy!r}")

        if is_uncertain:
            uncertain.append(ImageUncertainty(
                image_path=str(img_path),
                label_path=str(lbl_path),
                score=score,
                num_boxes=len(boxes),
                min_confidence=min_conf,
                mean_confidence=mean_conf,
            ))

    # Sort: ascending for conf strategies, descending for entropy
    reverse = (strategy == "entropy")
    uncertain.sort(key=lambda x: x.score, reverse=reverse)

    if top_k is not None:
        uncertain = uncertain[:top_k]

    logger.info(
        f"[UncertaintySampler] strategy={strategy}, "
        f"threshold={threshold}, found {len(uncertain)} uncertain images"
    )
    return uncertain
