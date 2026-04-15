"""
ChromaDB vector store wrapper for the RAG pipeline.

Handles document storage, embedding indexing, and similarity search.
"""

import chromadb


class VectorStore:
    """ChromaDB wrapper for document storage and retrieval."""

    def __init__(
        self,
        collection_name: str = "blys_knowledge",
        persist_dir: str = "./chroma_db",
    ):
        self.persist_dir = persist_dir
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
    ):
        """
        Add document chunks with their embeddings to the store.

        Args:
            chunks: List of {"chunk_id": str, "content": str, "source": str, "metadata": dict}
            embeddings: Corresponding embedding vectors.
        """
        self.collection.upsert(
            ids=[c["chunk_id"] for c in chunks],
            documents=[c["content"] for c in chunks],
            embeddings=embeddings,
            metadatas=[
                {"source": c["source"], "chunk_chars": c["metadata"]["chunk_chars"]}
                for c in chunks
            ],
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Vector similarity search.

        Returns:
            List of {"chunk_id": str, "content": str, "source": str, "score": float}
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for i in range(len(results["ids"][0])):
            hits.append(
                {
                    "chunk_id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "source": results["metadatas"][0][i]["source"],
                    "score": 1 - results["distances"][0][i],  # cosine similarity
                }
            )
        return hits

    @property
    def count(self) -> int:
        """Number of documents in the collection."""
        return self.collection.count()

    def reset(self):
        """Delete and recreate the collection."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"},
        )
