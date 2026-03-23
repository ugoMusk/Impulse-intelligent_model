from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import threading
import logging

from inference.pipeline import InferencePipeline
from model.model_loader import ModelLoader
from memory.memory_manager import MemoryManager


logger = logging.getLogger(__name__)


# =========================
# Request Schema
# =========================
class InferenceRequest(BaseModel):
    instruction: str = Field(..., min_length=1)
    context: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = None

    max_length: Optional[int] = Field(default=100, ge=1, le=512)

    temperature: Optional[float] = Field(default=1.0, ge=0.1, le=2.0)
    top_k: Optional[int] = Field(default=None, ge=1, le=100)
    top_p: Optional[float] = Field(default=None, ge=0.1, le=1.0)


# =========================
# Singleton Pipeline Manager
# =========================
class PipelineManager:
    _instance: Optional[InferencePipeline] = None
    _lock = threading.Lock()

    @classmethod
    def get_pipeline(cls) -> InferencePipeline:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    loader = ModelLoader(
                        config_path="configs/model_config.yaml",
                        model_dir="checkpoints/",
                        tokenizer_path="backend/model/tokenizer.model"
                    )
                    loader.load_model()

                    # ✅ Attach memory
                    memory = MemoryManager()

                    cls._instance = InferencePipeline(
                        model_loader=loader,
                        memory=memory
                    )

        return cls._instance


# =========================
# Router
# =========================
router = APIRouter()


@router.post("/generate")
def generate_text(request: InferenceRequest):
    try:
        pipeline = PipelineManager.get_pipeline()

        logger.info(f"Request: {request.instruction[:50]}")

        result = pipeline.run(
            instruction=request.instruction,
            context=request.context,
            history=request.history,
            max_length=request.max_length,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p
        )

        return result

    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# Streaming Endpoint (FIXED)
# =========================
@router.post("/stream")
def stream_text(request: InferenceRequest):
    try:
        pipeline = PipelineManager.get_pipeline()

        def event_stream():
            for token in pipeline.stream(
                instruction=request.instruction,
                context=request.context,
                history=request.history,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p
            ):
                yield f"data: {token}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream"
        )

    except Exception as e:
        logger.error(f"Streaming error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# Health Check
# =========================
@router.get("/health")
def health_check():
    return {"status": "ok"}