import logging
from typing import Dict

from edge_impulse.config import EdgeImpulseConfig
from edge_impulse.ingestion import EdgeImpulseIngestion
from edge_impulse.training import EdgeImpulseTraining

logger = logging.getLogger(__name__)


class EdgeImpulsePipeline:
    """
    Full integration layer:
    - ingestion
    - feedback loop
    - retraining trigger
    """

    def __init__(self):
        self.config = EdgeImpulseConfig()

        self.ingestion = EdgeImpulseIngestion(self.config)
        self.training = EdgeImpulseTraining(self.config)

    # =========================
    # MEMORY HOOK
    # =========================
    def ingest_memory(self, text: str):
        try:
            self.ingestion.ingest({
                "type": "memory",
                "text": text
            })
        except Exception as e:
            logger.error(f"Memory ingestion failed: {e}")

    # =========================
    # FEEDBACK HOOK
    # =========================
    def ingest_feedback(self, feedback: Dict):
        try:
            self.ingestion.ingest({
                "type": "feedback",
                "data": feedback
            })
        except Exception as e:
            logger.error(f"Feedback ingestion failed: {e}")

    # =========================
    # EVALUATION HOOK
    # =========================
    def handle_evaluation(self, eval_result: Dict):
        score = eval_result.get("quality_score", 1.0)

        if score < 0.6:
            logger.warning("Low quality detected → triggering retraining")

            try:
                self.training.trigger_training()
            except Exception as e:
                logger.error(f"Training trigger failed: {e}")