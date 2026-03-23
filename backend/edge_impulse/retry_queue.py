import queue
import threading
import time
import logging

logger = logging.getLogger(__name__)


class RetryQueue:
    """
    Handles async retry for failed ingestion
    """

    def __init__(self, worker, max_retries=3):
        self.q = queue.Queue()
        self.worker = worker
        self.max_retries = max_retries

        thread = threading.Thread(target=self._process)
        thread.daemon = True
        thread.start()

    def add(self, item):
        self.q.put((item, 0))

    def _process(self):
        while True:
            item, retries = self.q.get()

            try:
                self.worker(item)
            except Exception as e:
                if retries < self.max_retries:
                    logger.warning(f"Retry {retries+1} failed: {e}")
                    time.sleep(2 ** retries)
                    self.q.put((item, retries + 1))
                else:
                    logger.error("Max retries exceeded")

            self.q.task_done()