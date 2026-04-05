from __future__ import annotations

from celery import Celery

from app.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "interview_monitoring",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

celery_app.conf.update(
    task_default_queue=settings.celery_queue,
    task_track_started=True,
    result_expires=300,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    imports=("app.services.monitoring_tasks",),
)
