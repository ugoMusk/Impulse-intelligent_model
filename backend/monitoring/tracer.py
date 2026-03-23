import uuid
import time


class Trace:
    """
    Tracks full lifecycle of a request
    """

    def __init__(self):
        self.trace_id = str(uuid.uuid4())
        self.start_time = time.time()
        self.steps = []

    def log_step(self, name: str):
        self.steps.append({
            "step": name,
            "timestamp": time.time()
        })

    def finalize(self):
        return {
            "trace_id": self.trace_id,
            "total_time": round(time.time() - self.start_time, 4),
            "steps": self.steps
        }