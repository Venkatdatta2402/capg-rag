"""BM25 sparse retrieval for keyword-based search."""

import structlog
from rank_bm25 import BM25Okapi

from src.models.retrieval import RetrievalResult

logger = structlog.get_logger()


class SparseRetriever:
    """BM25 keyword-based retrieval over indexed chunks."""

    def __init__(self):
        self._corpus: list[dict] = []  # List of {chunk_id, text, source, ...}
        self._bm25: BM25Okapi | None = None

    def index(self, chunks: list[dict]) -> None:
        """Build the BM25 index from a list of chunk dicts.

        Each dict must have at least: chunk_id, text.
        """
        self._corpus = chunks
        tokenized = [doc["text"].lower().split() for doc in chunks]
        self._bm25 = BM25Okapi(tokenized)
        logger.info("sparse_retriever.indexed", num_chunks=len(chunks))

    def search(self, query: str, top_k: int = 20) -> list[RetrievalResult]:
        """Search the BM25 index."""
        if not self._bm25:
            logger.warning("sparse_retriever.not_indexed")
            return []

        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)

        scored_docs = sorted(
            zip(self._corpus, scores), key=lambda x: x[1], reverse=True
        )[:top_k]

        return [
            RetrievalResult(
                chunk_id=doc["chunk_id"],
                text=doc["text"],
                source=doc.get("source", ""),
                chapter=doc.get("chapter", ""),
                section=doc.get("section", ""),
                score=float(score),
                metadata=doc,
            )
            for doc, score in scored_docs
            if score > 0
        ]
