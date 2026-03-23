import os


class EdgeImpulseConfig:
    """
    Central configuration for Edge Impulse integration
    """

    def __init__(self):
        self.project_id = os.getenv("EI_PROJECT_ID")
        self.api_key = os.getenv("EI_API_KEY")

        self.ingestion_url = f"https://ingestion.edgeimpulse.com/api/{self.project_id}/data"
        self.studio_url = f"https://studio.edgeimpulse.com/v1/api/{self.project_id}"

        self.timeout = int(os.getenv("EI_TIMEOUT", 10))

        # Retry + batching
        self.max_retries = int(os.getenv("EI_MAX_RETRIES", 3))
        self.batch_size = int(os.getenv("EI_BATCH_SIZE", 10))

        # Async queue
        self.queue_enabled = os.getenv("EI_QUEUE_ENABLED", "true") == "true"