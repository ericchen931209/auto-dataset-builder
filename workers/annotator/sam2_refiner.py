"""
SAM2 bounding-box refinement.

Stage 2 of the three-stage annotation pipeline.
Takes YOLO bbox proposals and refines them using SAM2 segmentation masks.
If SAM2 is not installed or no GPU is available, falls back to the original YOLO bboxes.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from workers.annotator.yolo_annotator import AnnotationResult, BoundingBox

logger = logging.getLogger(__name__)


@dataclass
class RefinedAnnotation:
    image_path: str
    boxes: list[BoundingBox] = field(default_factory=list)
    refined_count: int = 0      # how many boxes were actually refined by SAM2
    fallback: bool = False       # True if SAM2 was unavailable


def _bbox_from_mask(mask: "np.ndarray", original: BoundingBox, img_h: int, img_w: int) -> BoundingBox:
    """Convert a binary mask to a tight-fitting YOLO normalized bbox."""
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return original

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h

    return BoundingBox(
        class_id=original.class_id,
        class_name=original.class_name,
        x_center=cx,
        y_center=cy,
        width=w,
        height=h,
        confidence=original.confidence,
    )


def _yolo_box_to_xyxy(box: BoundingBox, img_h: int, img_w: int) -> list[int]:
    """Convert YOLO normalized [cx,cy,w,h] to pixel [x1,y1,x2,y2]."""
    cx = box.x_center * img_w
    cy = box.y_center * img_h
    bw = box.width    * img_w
    bh = box.height   * img_h
    return [
        max(0, int(cx - bw / 2)),
        max(0, int(cy - bh / 2)),
        min(img_w, int(cx + bw / 2)),
        min(img_h, int(cy + bh / 2)),
    ]


def _try_load_sam2(checkpoint: str, config: str):
    """Return (predictor, device) or raise ImportError/RuntimeError."""
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam2(config, checkpoint, device=device)
    predictor = SAM2ImagePredictor(model)
    return predictor, device


def refine_with_sam2(
    annotation_results: list[AnnotationResult],
    checkpoint: str = "checkpoints/sam2_hiera_tiny.pt",
    config: str = "sam2_hiera_t.yaml",
    min_mask_area: int = 100,
    skip: bool = False,
) -> list[RefinedAnnotation]:
    """
    For each AnnotationResult, use SAM2 to refine each bounding box.

    Falls back to original YOLO bboxes when:
    - SAM2 / PyTorch is not installed
    - No valid mask is returned for a box
    - The image cannot be read

    Args:
        annotation_results: Output from run_yolo_batch().
        checkpoint: Path to SAM2 model checkpoint (.pt file).
        config: SAM2 model config name (tiny / small / base_plus / large).
        min_mask_area: Masks smaller than this pixel count are discarded.
        skip: If True, skip SAM2 entirely and pass YOLO boxes through
              unchanged (for ablation studies).

    Returns:
        List of RefinedAnnotation, one per input AnnotationResult.
    """
    if skip:
        return [
            RefinedAnnotation(image_path=ann.image_path, boxes=ann.boxes, fallback=True)
            for ann in annotation_results
        ]

    try:
        import cv2
        predictor, device = _try_load_sam2(checkpoint, config)
        sam2_available = True
        logger.info(f"SAM2 loaded on {device}")
    except (ImportError, Exception) as e:
        logger.warning(f"SAM2 unavailable ({e}), using YOLO bboxes as-is")
        sam2_available = False

    refined_results: list[RefinedAnnotation] = []

    for ann in annotation_results:
        if not ann.success or not ann.boxes:
            refined_results.append(RefinedAnnotation(
                image_path=ann.image_path,
                boxes=ann.boxes,
                fallback=True,
            ))
            continue

        if not sam2_available:
            refined_results.append(RefinedAnnotation(
                image_path=ann.image_path,
                boxes=ann.boxes,
                fallback=True,
            ))
            continue

        import cv2
        img_bgr = cv2.imread(ann.image_path)
        if img_bgr is None:
            logger.warning(f"Cannot read {ann.image_path}, skipping SAM2")
            refined_results.append(RefinedAnnotation(
                image_path=ann.image_path,
                boxes=ann.boxes,
                fallback=True,
            ))
            continue

        img_h, img_w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        try:
            predictor.set_image(img_rgb)
        except Exception as e:
            logger.warning(f"SAM2 set_image failed on {ann.image_path}: {e}")
            refined_results.append(RefinedAnnotation(
                image_path=ann.image_path,
                boxes=ann.boxes,
                fallback=True,
            ))
            continue

        refined_boxes: list[BoundingBox] = []
        refined_count = 0

        for box in ann.boxes:
            xyxy = _yolo_box_to_xyxy(box, img_h, img_w)
            input_box = np.array(xyxy, dtype=np.float32)

            try:
                import torch
                masks, scores, _ = predictor.predict(
                    box=input_box[None, :],
                    multimask_output=False,
                )
                # masks shape: (N, H, W) — take highest-score mask
                best_mask = masks[np.argmax(scores)]
                if best_mask.sum() >= min_mask_area:
                    refined_box = _bbox_from_mask(best_mask, box, img_h, img_w)
                    refined_boxes.append(refined_box)
                    refined_count += 1
                else:
                    refined_boxes.append(box)
            except Exception as e:
                logger.debug(f"SAM2 predict failed for one box: {e}")
                refined_boxes.append(box)

        refined_results.append(RefinedAnnotation(
            image_path=ann.image_path,
            boxes=refined_boxes,
            refined_count=refined_count,
            fallback=False,
        ))

    return refined_results
