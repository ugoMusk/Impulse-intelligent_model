import requests
from typing import Dict


class EdgeImpulseClient:
    """
    Handles all HTTP communication with Edge Impulse APIs
    """

    def __init__(self, config):
        self.config = config

        self.headers = {
            "x-api-key": config.api_key,
            "Content-Type": "application/json"
        }

    def post(self, url: str, payload: Dict):
        response = requests.post(
            url,
            json=payload,
            headers=self.headers,
            timeout=self.config.timeout
        )
        response.raise_for_status()
        return response.json()

    def get(self, url: str):
        response = requests.get(
            url,
            headers=self.headers,
            timeout=self.config.timeout
        )
        response.raise_for_status()
        return response.json()