from dataclasses import dataclass
from typing import Optional
import uuid
import time


@dataclass
class MemoryItem:
    id: str
    text: str
    embedding: list
    timestamp: float
    metadata: Optional[dict] = None

    @staticmethod
    def create(text: str, embedding: list, metadata: dict = None):
        return MemoryItem(
            id=str(uuid.uuid4()),
            text=text,
            embedding=embedding,
            timestamp=time.time(),
            metadata=metadata or {}
        )