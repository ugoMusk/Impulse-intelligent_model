from typing import Dict
import logging

from edge_impulse.client import EdgeImpulseClient
from edge_impulse.retry_queue import RetryQueue

logger = logging.getLogger(__name__)


class EdgeImpulseIngestion:
    """
    Handles data ingestion into Edge Impulse
    """

    def __init__(self, config):
        self.config = config
        self.client = EdgeImpulseClient(config)

        self.queue = RetryQueue(self._send) if config.queue_enabled else None

    def ingest(self, data: Dict):
        if self.queue:
            self.queue.add(data)
        else:
            self._send(data)

    def _send(self, data: Dict):
        payload = {
            "data": data
        }

        self.client.post(self.config.ingestion_url, payload)
        logger.info("Data ingested to Edge Impulse")