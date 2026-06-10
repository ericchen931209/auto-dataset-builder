import logging

from workers.annotator.yolo_annotator import BoundingBox, AnnotationResult

logger = logging.getLogger(__name__)


def run_yolo_world_batch(
    image_paths: list[str],
    class_names: list[str],
    model_path: str = "yolov8s-worldv2.pt",
    confidence_threshold: float = 0.25,
) -> list[AnnotationResult]:
    """
    Run YOLO-World open-vocabulary inference on a batch of images.

    Unlike run_yolo_batch (Stage 1's COCO-80 pretrained YOLOv11), class_names
    here are arbitrary text prompts — the detector is not restricted to any
    fixed vocabulary. Returns AnnotationResult per image, filtered by
    confidence_threshold.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed — cannot run YOLO-World annotation")
        return [AnnotationResult(p, success=False, error="ultralytics not installed")
                for p in image_paths]

    if not class_names:
        return [AnnotationResult(p, success=False, error="class_names is empty")
                for p in image_paths]

    model = YOLO(model_path)
    model.set_classes(class_names)

    results: list[AnnotationResult] = []

    for image_path in image_paths:
        try:
            preds = model(image_path, conf=confidence_threshold, verbose=False)
            boxes: list[BoundingBox] = []

            for pred in preds:
                for box in pred.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = class_names[cls_id]

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
            logger.warning(f"YOLO-World failed on {image_path}: {e}")
            results.append(AnnotationResult(image_path=image_path, success=False, error=str(e)))

    return results
