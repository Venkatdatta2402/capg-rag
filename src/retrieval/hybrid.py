"""Hybrid retrieval combining dense (vector) and sparse (BM25) results via RRF."""

import structlog

from src.models.retrieval import RetrievalResult
from src.retrieval.sparse import SparseRetriever
from src.retrieval.vector_store import VectorStore

logger = structlog.get_logger()


class HybridRetriever:
    """Reciprocal Rank Fusion of dense and sparse retrieval results.

    Used for the initial retrieval pass (before query expansion).
    grade + subject are always applied to scope results to the correct partition.
    BM25 results are post-filtered by grade/subject from chunk metadata.
    """

    def __init__(self, vector_store: VectorStore, sparse_retriever: SparseRetriever):
        self._dense = vector_store
        self._sparse = sparse_retriever

    async def search(
        self,
        query: str,
        grade: str = "",
        subject: str = "",
        top_k: int = 20,
        rrf_k: int = 60,
    ) -> list[RetrievalResult]:
        """Run both dense and sparse retrieval, fuse with RRF.

        Dense results are filtered by grade+subject at the Qdrant level.
        Sparse results are post-filtered by grade+subject from payload metadata.
        """
        dense_results = await self._dense.search(
            query=query, grade=grade, subject=subject, top_k=top_k * 2
        )
        sparse_candidates = self._sparse.search(query, top_k=top_k * 2)

        # Post-filter sparse by grade/subject when scoping is required
        if grade or subject:
            sparse_results = [
                r for r in sparse_candidates
                if (not grade or r.metadata.get("grade", "") == grade)
                and (not subject or r.metadata.get("subject", "") == subject)
            ]
        else:
            sparse_results = sparse_candidates

        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, RetrievalResult] = {}

        for rank, result in enumerate(dense_results):
            rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0) + 1 / (rrf_k + rank + 1)
            chunk_map[result.chunk_id] = result

        for rank, result in enumerate(sparse_results):
            rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0) + 1 / (rrf_k + rank + 1)
            if result.chunk_id not in chunk_map:
                chunk_map[result.chunk_id] = result

        sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]

        results = []
        for cid in sorted_ids:
            result = chunk_map[cid]
            result.score = rrf_scores[cid]
            results.append(result)

        logger.debug(
            "hybrid_retriever.search",
            grade=grade,
            subject=subject,
            dense=len(dense_results),
            sparse=len(sparse_results),
            fused=len(results),
        )
        return results
