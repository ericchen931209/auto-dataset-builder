import os
import logging
from workers.celery_app import celery_app
from workers.cleaner.cleaning_pipeline import clean_dataset

logger = logging.getLogger(__name__)

STORAGE_PATH = os.getenv("DATASET_STORAGE_PATH", "/app/datasets")


@celery_app.task(bind=True, name="cleaner.clean_dataset")
def run_clean_dataset(
    self,
    dataset_id: int,
    blur_threshold: float = 100.0,
    dark_threshold: float = 20.0,
    bright_threshold: float = 0.98,
    dry_run: bool = False,
):
    """Remove blurry, dark, overexposed, and too-small images from a dataset."""
    image_dir = os.path.join(STORAGE_PATH, str(dataset_id), "frames")
    labels_dir = os.path.join(STORAGE_PATH, str(dataset_id), "labels")

    self.update_state(state="PROGRESS", meta={
        "progress": 0.1,
        "message": "Analyzing image quality...",
    })

    report = clean_dataset(
        image_dir=image_dir,
        labels_dir=labels_dir if os.path.exists(labels_dir) else None,
        blur_threshold=blur_threshold,
        dark_threshold=dark_threshold,
        bright_threshold=bright_threshold,
        dry_run=dry_run,
    )

    logger.info(
        f"[dataset {dataset_id}] Cleaned: {report.kept} kept, {report.removed} removed "
        f"(blur={report.removed_blurry}, dark={report.removed_dark}, "
        f"overexp={report.removed_overexposed}, small={report.removed_too_small})"
    )

    return {
        "dataset_id": dataset_id,
        "dry_run": dry_run,
        "total": report.total,
        "kept": report.kept,
        "removed": report.removed,
        "breakdown": {
            "blurry": report.removed_blurry,
            "dark": report.removed_dark,
            "overexposed": report.removed_overexposed,
            "too_small": report.removed_too_small,
        },
    }
