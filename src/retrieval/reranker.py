"""Cross-encoder reranking for retrieval results."""

import structlog
from sentence_transformers import CrossEncoder

from config.settings import settings
from src.models.retrieval import RerankedChunk, RetrievalResult

logger = structlog.get_logger()


class Reranker:
    """Reranks retrieval candidates using a cross-encoder model.

    Cross-encoders perform pairwise relevance scoring (query, chunk),
    producing more accurate relevance scores than bi-encoder similarity.
    Latency scales linearly with the number of candidates.
    """

    def __init__(self):
        self._model = CrossEncoder(settings.reranker_model)

    def rerank(
        self, query: str, candidates: list[RetrievalResult], top_k: int | None = None
    ) -> list[RerankedChunk]:
        """Rerank candidates and return the top-k.

        Args:
            query: The enriched query.
            candidates: Retrieved chunks to rerank.
            top_k: Number of chunks to keep after reranking.
        """
        top_k = top_k or settings.top_k_rerank

        if not candidates:
            return []

        pairs = [(query, c.text) for c in candidates]
        scores = self._model.predict(pairs)

        scored = sorted(
            zip(candidates, scores), key=lambda x: x[1], reverse=True
        )[:top_k]

        reranked = [
            RerankedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                source=chunk.source,
                original_score=chunk.score,
                rerank_score=float(score),
                metadata=chunk.metadata,
            )
            for chunk, score in scored
        ]

        logger.debug(
            "reranker.done",
            input_count=len(candidates),
            output_count=len(reranked),
            top_score=reranked[0].rerank_score if reranked else 0,
        )
        return reranked
