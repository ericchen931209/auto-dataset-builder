import os
import logging
from workers.celery_app import celery_app
from workers.extractor.frame_extractor import extract_fixed_rate, extract_adaptive

logger = logging.getLogger(__name__)

STORAGE_PATH = os.getenv("DATASET_STORAGE_PATH", "/app/datasets")


@celery_app.task(bind=True, name="extractor.extract_frames")
def extract_frames(
    self,
    dataset_id: int,
    video_path: str,
    mode: str = "adaptive",
    fps: float = 1.0,
    ssim_threshold: float = 0.85,
):
    """
    Extract frames from a video file.
    mode: "fixed" (constant fps) or "adaptive" (scene-change detection via SSIM)
    """
    output_dir = os.path.join(STORAGE_PATH, str(dataset_id), "frames")
    self.update_state(state="PROGRESS", meta={"progress": 0.1, "message": f"Extracting frames ({mode})..."})

    if mode == "adaptive":
        result = extract_adaptive(video_path, output_dir, ssim_threshold=ssim_threshold)
    else:
        result = extract_fixed_rate(video_path, output_dir, fps=fps)

    logger.info(
        f"[dataset {dataset_id}] Extracted {result.extracted_frames} frames "
        f"from {os.path.basename(video_path)}"
    )

    return {
        "dataset_id": dataset_id,
        "video_path": video_path,
        "mode": mode,
        "total_frames": result.total_frames,
        "extracted_frames": result.extracted_frames,
        "output_dir": output_dir,
        "frame_paths": result.frame_paths,
    }
