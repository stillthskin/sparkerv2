import os
import django
from celery import Celery
from celery.signals import worker_process_init
from celery.signals import worker_ready
import logging

logger = logging.getLogger(__name__)

# Set the default Django settings module
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'sparkerv2.settings')

# Initialize Django BEFORE using any Django components like cache
django.setup()

app = Celery('sparkerv2')

app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()


@worker_ready.connect
def clear_startup_lock(sender, **kwargs):
    from django.core.cache import cache
    from sparkermain.tasks import connect_websocket_task

    lock_key = "startup_websocket_task_lock"
    cache.delete(lock_key)
    logger.info("Cleared cache lock key at startup.")

    logger.info("Scheduling WebSocket task...")
    connect_websocket_task.apply_async(queue='websocket_queue')


