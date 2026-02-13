import os
from celery import Celery
from celery.signals import worker_process_init
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
    worker_hijack_root_logger=False,  # Don't let Celery override our logging
)


@worker_process_init.connect
def configure_worker_logging(**kwargs):
    """
    Re-configure logging in each forked Celery worker process.

    Without this, forked workers lose the file handlers and log config
    from the parent process, so app.worker / app.document_processing
    logs silently disappear.
    """
    from app.logging_config import setup_logging
    setup_logging()


# Also configure for the main Celery process (beat, inspect, etc.)
from app.logging_config import setup_logging
setup_logging()
