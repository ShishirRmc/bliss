"""
Embedding model wrapper for the RAG pipeline.

Uses sentence-transformers (local, CPU, no API cost).
"""

import os
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Local embedding model for document and query encoding."""

    def __init__(self, model_name: str | None = None):
        model_name = model_name or os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Encode a list of texts into embedding vectors."""
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Encode a single query string."""
        return self.embed([query])[0]

    def __repr__(self) -> str:
        return f"EmbeddingModel(model='{self.model_name}', dim={self.dimension})"
