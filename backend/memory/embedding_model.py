import numpy as np
import tensorflow as tf


class EmbeddingModel:
    """
    Lightweight embedding generator.
    Replace with SentenceTransformers or OpenAI embeddings in production.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def encode(self, text: str) -> np.ndarray:
        """
        Dummy embedding (replace later with real model)
        """
        hash_val = abs(hash(text)) % (10 ** 8)
        np.random.seed(hash_val)
        return np.random.rand(self.dim).astype("float32")