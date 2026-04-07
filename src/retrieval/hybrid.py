"""Hybrid retrieval combining dense (vector) and sparse (BM25) results."""

import structlog

from src.models.retrieval import RetrievalResult
from src.retrieval.vector_store import VectorStore
from src.retrieval.sparse import SparseRetriever

logger = structlog.get_logger()


class HybridRetriever:
    """Reciprocal Rank Fusion of dense and sparse retrieval results."""

    def __init__(self, vector_store: VectorStore, sparse_retriever: SparseRetriever):
        self._dense = vector_store
        self._sparse = sparse_retriever

    async def search(self, query: str, top_k: int = 20, rrf_k: int = 60) -> list[RetrievalResult]:
        """Run both dense and sparse retrieval, fuse with RRF.

        Args:
            query: The search query.
            top_k: Number of results to return after fusion.
            rrf_k: RRF constant (default 60).
        """
        dense_results = await self._dense.search(query, top_k=top_k * 2)
        sparse_results = self._sparse.search(query, top_k=top_k * 2)

        # Compute RRF scores
        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, RetrievalResult] = {}

        for rank, result in enumerate(dense_results):
            rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0) + 1 / (rrf_k + rank + 1)
            chunk_map[result.chunk_id] = result

        for rank, result in enumerate(sparse_results):
            rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0) + 1 / (rrf_k + rank + 1)
            if result.chunk_id not in chunk_map:
                chunk_map[result.chunk_id] = result

        # Sort by fused score
        sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]

        results = []
        for cid in sorted_ids:
            result = chunk_map[cid]
            result.score = rrf_scores[cid]
            results.append(result)

        logger.debug("hybrid_retriever.search", query_len=len(query), results=len(results))
        return results
