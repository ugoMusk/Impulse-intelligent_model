import time
from typing import Optional, Dict, List

from model.model_loader import ModelLoader
from inference.prompt_builder import PromptBuilder
from inference.output_formatter import OutputFormatter

# ✅ Monitoring
from monitoring.metrics import MetricsCollector
from monitoring.tracer import Trace
from monitoring.evaluator import Evaluator
from monitoring.logger import logger


class InferencePipeline:
    """
    Fully production-grade inference pipeline
    """

    def __init__(self, model_loader: ModelLoader, memory=None):
        self.model_loader = model_loader
        self.prompt_builder = PromptBuilder()
        self.formatter = OutputFormatter()
        self.memory = memory

    # =========================
    # INPUT SANITIZATION
    # =========================
    def _sanitize(self, text: str) -> str:
        return text.strip()

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

        metrics = MetricsCollector()
        trace = Trace()
        evaluator = Evaluator()

        trace.log_step("start")

        # =========================
        # Validation
        # =========================
        instruction = self._sanitize(instruction)

        if top_k is not None and top_p is not None:
            raise ValueError("Use either top_k OR top_p, not both.")

        # =========================
        # Memory Retrieval
        # =========================
        if self.memory:
            trace.log_step("memory_retrieval")
            retrieved_context = self.memory.retrieve(instruction)
            context = f"{retrieved_context}\n{context or ''}"

        # =========================
        # Prompt Build
        # =========================
        trace.log_step("prompt_building")
        prompt = self.prompt_builder.build(
            instruction=instruction,
            context=context,
            history=history
        )

        # =========================
        # Generation
        # =========================
        trace.log_step("generation")
        raw_output = self.model_loader.generate(
            prompt=prompt,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )

        metrics.add_tokens(len(raw_output.split()))

        # =========================
        # Formatting
        # =========================
        trace.log_step("formatting")
        formatted = self.formatter.format_response(raw_output)

        # =========================
        # Evaluation
        # =========================
        trace.log_step("evaluation")
        eval_result = evaluator.evaluate(formatted["answer"])

        # =========================
        # Memory Write
        # =========================
        if self.memory:
            trace.log_step("memory_write")
            self.memory.add(f"User: {instruction}")
            self.memory.add(f"Assistant: {formatted['answer']}")

        trace.log_step("end")

        return {
            "trace": trace.finalize(),
            "metrics": metrics.finalize(),
            "evaluation": eval_result,
            **formatted
        }

    # =========================
    # STREAMING (FIXED)
    # =========================
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
        Production streaming with memory + accumulation
        """

        instruction = self._sanitize(instruction)

        if top_k is not None and top_p is not None:
            raise ValueError("Use either top_k OR top_p, not both.")

        if self.memory:
            retrieved_context = self.memory.retrieve(instruction)
            context = f"{retrieved_context}\n{context or ''}"

        prompt = self.prompt_builder.build(instruction, context, history)

        full_output = ""

        for token in self.model_loader.stream_generate(
            prompt,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        ):
            full_output += token
            yield token

        # ✅ After stream ends → store memory
        if self.memory:
            formatted = self.formatter.format_response(full_output)
            self.memory.add(f"User: {instruction}")
            self.memory.add(f"Assistant: {formatted['answer']}")