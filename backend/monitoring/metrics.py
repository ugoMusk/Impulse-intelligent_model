import time
from typing import Dict


class MetricsCollector:
    """
    Tracks performance + usage metrics
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.start_time = time.time()
        self.tokens_generated = 0

    def add_tokens(self, count: int):
        self.tokens_generated += count

    def finalize(self) -> Dict:
        latency = time.time() - self.start_time

        return {
            "latency": round(latency, 4),
            "tokens_generated": self.tokens_generated,
            "tokens_per_second": round(
                self.tokens_generated / latency, 2
            ) if latency > 0 else 0
        }