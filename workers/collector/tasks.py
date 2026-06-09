import os
import logging
from workers.celery_app import celery_app
from workers.collector.youtube_downloader import download_videos
from workers.collector.image_searcher import expand_keywords, search_and_download_images
from workers.collector.deduplicator import remove_duplicates

logger = logging.getLogger(__name__)

STORAGE_PATH = os.getenv("DATASET_STORAGE_PATH", "/app/datasets")


@celery_app.task(bind=True, name="collector.collect_youtube")
def collect_youtube(
    self,
    dataset_id: int,
    keywords: list[str],
    region: str = "",
    max_videos: int = 5,
    resolution: str = "720",
):
    """
    Download YouTube videos for a dataset.
    Updates Celery task state with progress so the Dashboard can poll it.
    """
    video_dir = os.path.join(STORAGE_PATH, str(dataset_id), "videos")
    expanded = expand_keywords(keywords, target=keywords[0], region=region)

    def on_progress(pct: float, msg: str):
        self.update_state(state="PROGRESS", meta={"progress": pct, "message": msg})

    results = download_videos(
        keywords=expanded,
        output_dir=video_dir,
        max_videos=max_videos,
        resolution=resolution,
        progress_callback=on_progress,
    )

    success = [r for r in results if r.success]
    logger.info(f"[dataset {dataset_id}] Downloaded {len(success)}/{len(results)} videos")

    return {
        "dataset_id": dataset_id,
        "total": len(results),
        "success": len(success),
        "video_dir": video_dir,
        "files": [r.output_path for r in success],
    }


@celery_app.task(bind=True, name="collector.collect_images")
def collect_images(
    self,
    dataset_id: int,
    keywords: list[str],
    region: str = "",
    max_images: int = 200,
):
    """
    Download images from Google Custom Search and deduplicate.
    """
    image_dir = os.path.join(STORAGE_PATH, str(dataset_id), "raw_images")
    expanded = expand_keywords(keywords, target=keywords[0], region=region)

    self.update_state(state="PROGRESS", meta={"progress": 0.1, "message": "Downloading images..."})

    results = search_and_download_images(
        keywords=expanded,
        output_dir=image_dir,
        max_images=max_images,
        google_api_key=os.getenv("GOOGLE_SEARCH_API_KEY", ""),
        google_cx=os.getenv("GOOGLE_SEARCH_CX", ""),
    )

    self.update_state(state="PROGRESS", meta={"progress": 0.8, "message": "Removing duplicates..."})
    dedup_result = remove_duplicates(image_dir)

    logger.info(
        f"[dataset {dataset_id}] Images: {dedup_result['kept']} kept, "
        f"{dedup_result['removed']} duplicates removed"
    )

    return {
        "dataset_id": dataset_id,
        "downloaded": len([r for r in results if r.success]),
        "after_dedup": dedup_result["kept"],
        "duplicates_removed": dedup_result["removed"],
        "image_dir": image_dir,
    }
