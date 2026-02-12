import os
from celery import Celery
from app.config import get_settings

settings = get_settings()

celery_app = Celery(
    "worker",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.worker"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
)

# Setup logging
from app.logging_config import setup_logging
setup_logging()
