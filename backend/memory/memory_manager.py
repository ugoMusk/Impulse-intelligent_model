import os
import pickle
from typing import List, Optional

from memory.embedding_model import EmbeddingModel
from memory.vector_store import VectorStore
from memory.retriever import Retriever


class MemoryManager:
    """
    Central memory system

    Features:
    - Store interactions
    - Retrieve relevant context
    - Persist to disk
    """

    def __init__(self, dim: int = 256, storage_path: str = "memory_store.pkl"):
        self.embedding_model = EmbeddingModel(dim)
        self.vector_store = VectorStore(dim)
        self.retriever = Retriever(self.vector_store, self.embedding_model)

        self.storage_path = storage_path

        self._load()

    # =========================
    # STORE MEMORY
    # =========================
    def add(self, text: str, metadata: Optional[dict] = None):
        embedding = self.embedding_model.encode(text)

        self.vector_store.add([embedding], [text])

        self._save()

    # =========================
    # RETRIEVE MEMORY
    # =========================
    def retrieve(self, query: str, top_k: int = 3) -> str:
        results = self.retriever.retrieve(query, top_k)

        return "\n".join(results)

    # =========================
    # PERSISTENCE
    # =========================
    def _save(self):
        with open(self.storage_path, "wb") as f:
            pickle.dump(self.vector_store.texts, f)

    def _load(self):
        if os.path.exists(self.storage_path):
            with open(self.storage_path, "rb") as f:
                texts = pickle.load(f)

            embeddings = [self.embedding_model.encode(t) for t in texts]
            self.vector_store.add(embeddings, texts)