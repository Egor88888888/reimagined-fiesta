"""
DocLens Celery Worker
For async document processing (optional — API also supports sync mode).
"""
from celery import Celery
from app.config import get_settings

settings = get_settings()

celery_app = Celery(
    "doclens",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_soft_time_limit=120,  # 2 min soft limit
    task_time_limit=180,       # 3 min hard limit
    worker_max_tasks_per_child=100,  # Restart worker after 100 tasks (memory)
)


@celery_app.task(name="recognize_document", bind=True, max_retries=2)
def recognize_document_task(self, image_bytes: bytes, document_type_hint: str = None) -> dict:
    """Async document recognition task."""
    try:
        from app.core.orchestrator import get_pipeline
        pipeline = get_pipeline()
        result = pipeline.process(image_bytes, document_type_hint=document_type_hint)
        return result.to_dict()
    except Exception as e:
        raise self.retry(exc=e, countdown=5)
