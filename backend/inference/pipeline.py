import time
from typing import Optional, Dict, List

from model.model_loader import ModelLoader
from inference.prompt_builder import PromptBuilder
from inference.output_formatter import OutputFormatter


class InferencePipeline:
    """
    Production-grade inference pipeline
    """

    def __init__(self, model_loader: ModelLoader, memory=None):
        self.model_loader = model_loader
        self.prompt_builder = PromptBuilder()
        self.formatter = OutputFormatter()
        self.memory = memory

    def run(
        self,
        instruction: str,
        context: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> Dict:

        start_time = time.time()

        # =========================
        # Validate decoding params
        # =========================
        if top_k is not None and top_p is not None:
            raise ValueError("Use either top_k OR top_p, not both.")

        # =========================
        # Memory retrieval
        # =========================
        if self.memory:
            retrieved_context = self.memory.retrieve(instruction)
            context = f"{retrieved_context}\n{context or ''}"

        # =========================
        # Build prompt
        # =========================
        prompt = self.prompt_builder.build(
            instruction=instruction,
            context=context,
            history=history
        )

        # =========================
        # Generate
        # =========================
        raw_output = self.model_loader.generate(
            prompt=prompt,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )

        # =========================
        # Format
        # =========================
        formatted = self.formatter.format_response(raw_output)

        # =========================
        # Memory write-back (CRITICAL)
        # =========================
        if self.memory:
            self.memory.add(f"User: {instruction}")
            self.memory.add(f"Assistant: {formatted['answer']}")

        latency = time.time() - start_time

        return {
            "prompt": prompt,
            "latency": round(latency, 3),
            **formatted
        }

    def stream(
        self,
        instruction: str,
        context: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ):
        """
        True streaming generator
        """

        if top_k is not None and top_p is not None:
            raise ValueError("Use either top_k OR top_p, not both.")

        if self.memory:
            retrieved_context = self.memory.retrieve(instruction)
            context = f"{retrieved_context}\n{context or ''}"

        prompt = self.prompt_builder.build(instruction, context, history)

        for token in self.model_loader.stream_generate(
            prompt,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        ):
            yield token