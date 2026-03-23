from typing import List


class Retriever:
    """
    Handles semantic search over memory
    """

    def __init__(self, vector_store, embedding_model):
        self.vector_store = vector_store
        self.embedding_model = embedding_model

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        query_embedding = self.embedding_model.encode(query)

        results = self.vector_store.search(query_embedding, top_k)

        return [text for text, _ in results]