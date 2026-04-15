"""
Hybrid retriever with BM25 + vector search and cross-encoder reranking.

Combines sparse (BM25) and dense (vector) retrieval using Reciprocal Rank
Fusion, then reranks the merged results with a cross-encoder model.
"""

import re
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from rag.embeddings import EmbeddingModel
from rag.vector_store import VectorStore


class HybridRetriever:
    """BM25 + Vector search with cross-encoder reranking."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
        chunks: list[dict],
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.chunks = chunks
        self.chunk_lookup = {c["chunk_id"]: c for c in chunks}

        # Build BM25 index
        tokenized_corpus = [self._tokenize(c["content"]) for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # Load cross-encoder reranker
        self.reranker = CrossEncoder(reranker_model)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Hybrid retrieval pipeline:
        1. BM25 search → top 20 candidates
        2. Vector search → top 20 candidates
        3. Reciprocal Rank Fusion (RRF) to merge
        4. Cross-encoder reranking on merged top candidates
        5. Return top_k results

        Returns:
            List of {"chunk_id", "content", "source", "score", "bm25_rank", "vector_rank"}
        """
        bm25_results = self._bm25_search(query, top_k=20)

        query_embedding = self.embedding_model.embed_query(query)
        vector_results = self.vector_store.search(query_embedding, top_k=20)

        merged = self._rrf_merge(bm25_results, vector_results, k=60)

        candidates = merged[: top_k * 3]
        reranked = self._rerank(query, candidates)

        return reranked[:top_k]

    def _bm25_search(self, query: str, top_k: int = 20) -> list[dict]:
        """BM25 sparse retrieval."""
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top_k indices
        top_indices = scores.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                chunk = self.chunks[idx]
                results.append(
                    {
                        "chunk_id": chunk["chunk_id"],
                        "content": chunk["content"],
                        "source": chunk["source"],
                        "score": float(scores[idx]),
                    }
                )
        return results

    def _rrf_merge(
        self,
        bm25_results: list[dict],
        vector_results: list[dict],
        k: int = 60,
    ) -> list[dict]:
        """
        Reciprocal Rank Fusion — merge two ranked lists.

        RRF score = Σ 1 / (k + rank_i) for each list containing the document.
        """
        scores: dict[str, float] = {}
        chunk_data: dict[str, dict] = {}
        bm25_ranks: dict[str, int] = {}
        vector_ranks: dict[str, int] = {}

        for rank, result in enumerate(bm25_results):
            cid = result["chunk_id"]
            scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
            chunk_data[cid] = result
            bm25_ranks[cid] = rank + 1

        for rank, result in enumerate(vector_results):
            cid = result["chunk_id"]
            scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
            chunk_data[cid] = result
            vector_ranks[cid] = rank + 1

        # Sort by RRF score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        merged = []
        for cid in sorted_ids:
            entry = chunk_data[cid].copy()
            entry["rrf_score"] = scores[cid]
            entry["bm25_rank"] = bm25_ranks.get(cid, -1)
            entry["vector_rank"] = vector_ranks.get(cid, -1)
            merged.append(entry)

        return merged

    def _rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """Rerank candidates using a cross-encoder model."""
        if not candidates:
            return []

        pairs = [(query, c["content"]) for c in candidates]
        scores = self.reranker.predict(pairs)

        for i, candidate in enumerate(candidates):
            candidate["rerank_score"] = float(scores[i])

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace + punctuation tokenizer for BM25."""
        text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        return tokens
