from celery import Celery
import os

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

celery_app = Celery(
    "adb_workers",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=[
        "workers.collector.tasks",
        "workers.extractor.tasks",
        "workers.annotator.tasks",
        "workers.cleaner.tasks",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Taipei",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,   # one task at a time per worker (GPU jobs)
)
