import json
import os
from typing import Dict


class FeedbackStore:
    """
    Stores user feedback for future training
    """

    def __init__(self, path: str = "feedback.jsonl"):
        self.path = path

    def save(self, data: Dict):
        with open(self.path, "a") as f:
            f.write(json.dumps(data) + "\n")