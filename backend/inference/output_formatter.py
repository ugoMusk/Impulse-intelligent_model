import re
from typing import Dict


class OutputFormatter:
    """
    Cleans and structures model outputs safely
    """

    def clean(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = text.replace("<pad>", "").replace("<unk>", "")
        return text.strip()

    def extract_output(self, text: str) -> str:
        if "Output:" in text:
            return text.split("Output:")[-1].strip()
        return text

    def validate(self, text: str) -> str:
        if not text or len(text.strip()) == 0:
            return "I'm sorry, I couldn't generate a response."
        return text

    def format_response(self, raw_text: str) -> Dict:
        cleaned = self.clean(raw_text)
        answer = self.extract_output(cleaned)
        answer = self.validate(answer)

        return {
            "raw_output": raw_text,
            "cleaned_output": cleaned,
            "answer": answer
        }
    