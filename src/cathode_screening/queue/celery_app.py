"""
Celery application for async prediction jobs.

Decouples the API from long-running inference by routing batch predictions
through a Redis-backed task queue. Workers run GPU inference independently.

Configuration:
    CATHODE_REDIS_URL=redis://localhost:6379/0
    CATHODE_CELERY_CONCURRENCY=2  (worker processes)

Usage:
    # Start worker
    celery -A cathode_screening.queue.celery_app worker --loglevel=info --concurrency=2

    # Start beat scheduler (for periodic drift checks)
    celery -A cathode_screening.queue.celery_app beat --loglevel=info
"""
from __future__ import annotations

import os

from celery import Celery

REDIS_URL = os.getenv("CATHODE_REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "cathode_screening",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.update(
    # Serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",

    # Timeouts
    task_soft_time_limit=120,   # 2 min soft limit
    task_time_limit=180,        # 3 min hard limit
    result_expires=3600,        # Results expire after 1 hour

    # Worker settings
    worker_prefetch_multiplier=1,  # One task at a time per worker (GPU bound)
    worker_max_tasks_per_child=100,  # Restart workers periodically (leak prevention)

    # Task routing
    task_routes={
        "cathode_screening.queue.tasks.predict_structure_task": {"queue": "inference"},
        "cathode_screening.queue.tasks.predict_batch_task": {"queue": "inference"},
        "cathode_screening.queue.tasks.compute_drift_task": {"queue": "monitoring"},
    },

    # Periodic tasks (beat schedule)
    beat_schedule={
        "drift-check-hourly": {
            "task": "cathode_screening.queue.tasks.compute_drift_task",
            "schedule": 3600.0,  # Every hour
            "kwargs": {},
        },
    },
)

# Auto-discover tasks
celery_app.autodiscover_tasks(["cathode_screening.queue"])
