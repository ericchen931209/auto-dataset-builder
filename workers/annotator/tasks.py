import os
import logging
from pathlib import Path
from workers.celery_app import celery_app
from workers.annotator.yolo_annotator import run_yolo_batch, save_yolo_labels

logger = logging.getLogger(__name__)

STORAGE_PATH = os.getenv("DATASET_STORAGE_PATH", "/app/datasets")
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov11n.pt")


@celery_app.task(bind=True, name="annotator.run_yolo")
def run_yolo_annotation(
    self,
    dataset_id: int,
    image_paths: list[str],
    target_classes: list[str],
    confidence_threshold: float = 0.5,
):
    """
    Stage 1 of annotation pipeline: run YOLOv11 on all images.
    Saves YOLO-format .txt label files next to the images.
    """
    labels_dir = os.path.join(STORAGE_PATH, str(dataset_id), "labels")
    total = len(image_paths)

    self.update_state(state="PROGRESS", meta={
        "progress": 0.05,
        "message": f"Running YOLO on {total} images...",
    })

    # Process in batches of 32 for memory efficiency
    batch_size = 32
    all_results = []

    for i in range(0, total, batch_size):
        batch = image_paths[i:i + batch_size]
        batch_results = run_yolo_batch(
            image_paths=batch,
            model_path=YOLO_MODEL,
            target_classes=target_classes,
            confidence_threshold=confidence_threshold,
        )
        all_results.extend(batch_results)

        progress = min(0.9, (i + len(batch)) / total)
        self.update_state(state="PROGRESS", meta={
            "progress": progress,
            "message": f"Annotated {i + len(batch)}/{total} images",
        })

    # Build class map (target_classes index)
    class_map = {cls: idx for idx, cls in enumerate(target_classes)}
    written = save_yolo_labels(all_results, labels_dir, class_map=class_map)

    annotated = sum(1 for r in all_results if r.success and r.boxes)
    empty = sum(1 for r in all_results if r.success and not r.boxes)

    logger.info(
        f"[dataset {dataset_id}] YOLO: {annotated} annotated, "
        f"{empty} empty (no detections), labels -> {labels_dir}"
    )

    return {
        "dataset_id": dataset_id,
        "total_images": total,
        "annotated_images": annotated,
        "empty_images": empty,
        "labels_dir": labels_dir,
        "label_files_written": len(written),
    }


@celery_app.task(bind=True, name="annotator.run_pipeline")
def run_annotation_pipeline(self, dataset_id: int, image_ids: list[int]):
    """
    Full three-stage pipeline: YOLO → SAM2 → LLM verification.
    SAM2 and LLM stages are stubs until V0.5.
    """
    # V0.3: delegates to YOLO only
    # V0.5: will chain SAM2 + LLM tasks
    return {"status": "use run_yolo_annotation for V0.3"}
