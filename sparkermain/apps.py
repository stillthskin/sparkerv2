import logging
from django.apps import AppConfig
from django.core.cache import cache
from .tasks import connect_websocket_task

logger = logging.getLogger(__name__)

class SparkermainConfig(AppConfig):
    name = 'sparkermain'

    def ready(self):
        lock_key = "startup_websocket_task_lock"
        # Only the first process that adds this key gets to schedule the task
          
        # if force:
        #     cache.delete(lock_key)
        if cache.add(lock_key, "locked", timeout=600):
            connect_websocket_task.apply_async(queue='websocket_queue')
            logger.info("WebSocket task scheduled ONCE on startup")
