import os
from celery import Celery
from app.config import get_settings

settings = get_settings()

celery_app = Celery(
    "worker",
    broker=settings.redis_url,
    backend=settings.redis_url
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
)

# Placeholder for task routes
celery_app.conf.task_routes = {
    "app.worker.process_document": "documents-queue",
}
