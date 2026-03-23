import logging

from edge_impulse.client import EdgeImpulseClient

logger = logging.getLogger(__name__)


class EdgeImpulseTraining:
    """
    Handles training triggers via Studio API
    """

    def __init__(self, config):
        self.client = EdgeImpulseClient(config)
        self.base_url = config.studio_url

    def trigger_training(self):
        url = f"{self.base_url}/impulses/train"
        response = self.client.post(url, {})
        logger.info("Training triggered")
        return response

    def get_training_status(self):
        url = f"{self.base_url}/training/status"
        return self.client.get(url)