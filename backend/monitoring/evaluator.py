from typing import Dict


class Evaluator:
    """
    Basic response quality checks
    """

    def evaluate(self, response: str) -> Dict:
        length = len(response.split())

        return {
            "length": length,
            "is_empty": length == 0,
            "quality_score": self._score(response)
        }

    def _score(self, text: str) -> float:
        if len(text.strip()) == 0:
            return 0.0

        if len(text.split()) < 3:
            return 0.3

        return 0.9