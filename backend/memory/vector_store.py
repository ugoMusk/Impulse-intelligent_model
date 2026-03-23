import faiss
import numpy as np
from typing import List, Tuple


class VectorStore:
    """
    FAISS-based vector storage
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []

    def add(self, embeddings: List[np.ndarray], texts: List[str]):
        vectors = np.vstack(embeddings)
        self.index.add(vectors)
        self.texts.extend(texts)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        query_embedding = np.expand_dims(query_embedding, axis=0)

        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.texts):
                results.append((self.texts[idx], float(distances[0][i])))

        return results