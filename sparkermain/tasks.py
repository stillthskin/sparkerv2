# sparkermain/tasks.py
import logging
from celery import shared_task
from .Model import Model

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)

@shared_task(bind=True, queue='websocket_queue')
def connect_websocket_task(self):
    logger.info("Starting WebSocket connection")

    try:
        model = Model('BTCUSDT', 0.00015)
        model.connect_to_websocket()
    except Exception as exc:
        logger.error(f"WebSocket connection failed: {exc}")
        raise self.retry(exc=exc, countdown=10)
