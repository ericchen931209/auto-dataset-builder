import os
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    class_id: int
    class_name: str
    x_center: float    # YOLO normalized [0,1]
    y_center: float
    width: float
    height: float
    confidence: float


@dataclass
class AnnotationResult:
    image_path: str
    boxes: list[BoundingBox] = field(default_factory=list)
    success: bool = True
    error: str = ""


def run_yolo_batch(
    image_paths: list[str],
    model_path: str = "yolov11n.pt",
    class_names: list[str] | None = None,
    confidence_threshold: float = 0.5,
    target_classes: list[str] | None = None,
) -> list[AnnotationResult]:
    """
    Run YOLOv11 inference on a batch of images.
    Returns AnnotationResult for each image, filtered by confidence and target_classes.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed — cannot run YOLO annotation")
        return [AnnotationResult(p, success=False, error="ultralytics not installed")
                for p in image_paths]

    model = YOLO(model_path)
    results: list[AnnotationResult] = []

    for image_path in image_paths:
        try:
            preds = model(image_path, conf=confidence_threshold, verbose=False)
            boxes: list[BoundingBox] = []

            for pred in preds:
                for box in pred.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = pred.names[cls_id]

                    if target_classes and cls_name not in target_classes:
                        continue

                    xywhn = box.xywhn[0].tolist()  # normalized cx, cy, w, h
                    boxes.append(BoundingBox(
                        class_id=cls_id,
                        class_name=cls_name,
                        x_center=xywhn[0],
                        y_center=xywhn[1],
                        width=xywhn[2],
                        height=xywhn[3],
                        confidence=float(box.conf[0]),
                    ))

            results.append(AnnotationResult(image_path=image_path, boxes=boxes))

        except Exception as e:
            logger.warning(f"YOLO failed on {image_path}: {e}")
            results.append(AnnotationResult(image_path=image_path, success=False, error=str(e)))

    return results


def save_yolo_labels(
    annotation_results: list[AnnotationResult],
    labels_dir: str,
    class_map: dict[str, int] | None = None,
) -> list[str]:
    """
    Write annotation results to YOLO .txt label files.
    Returns list of written label file paths.
    """
    Path(labels_dir).mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    for result in annotation_results:
        if not result.success or not result.boxes:
            continue

        stem = Path(result.image_path).stem
        label_path = os.path.join(labels_dir, f"{stem}.txt")

        with open(label_path, "w") as f:
            for box in result.boxes:
                cls_id = class_map[box.class_name] if class_map else box.class_id
                f.write(
                    f"{cls_id} {box.x_center:.6f} {box.y_center:.6f} "
                    f"{box.width:.6f} {box.height:.6f}\n"
                )

        written.append(label_path)

    return written


def generate_dataset_yaml(
    dataset_dir: str,
    class_names: list[str],
    output_path: str,
) -> str:
    """Generate a YOLO dataset.yaml file."""
    content = f"""path: {dataset_dir}
train: images/train
val: images/val
test: images/test

nc: {len(class_names)}
names: {class_names}
"""
    with open(output_path, "w") as f:
        f.write(content)
    return output_path
