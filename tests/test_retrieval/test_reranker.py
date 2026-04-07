"""Tests for retrieval components."""

import pytest

from src.models.retrieval import RetrievalResult
from src.retrieval.sparse import SparseRetriever


class TestSparseRetriever:
    """Tests for BM25 sparse retrieval."""

    def test_index_and_search(self):
        retriever = SparseRetriever()
        chunks = [
            {"chunk_id": "c1", "text": "metre centimetre conversion length measurement"},
            {"chunk_id": "c2", "text": "addition subtraction basic arithmetic numbers"},
            {"chunk_id": "c3", "text": "converting metres to centimetres multiply by 100"},
        ]
        retriever.index(chunks)

        results = retriever.search("metre to centimetre conversion", top_k=2)
        assert len(results) <= 2
        # The most relevant chunks should rank higher
        chunk_ids = [r.chunk_id for r in results]
        assert "c1" in chunk_ids or "c3" in chunk_ids

    def test_search_without_index_returns_empty(self):
        retriever = SparseRetriever()
        results = retriever.search("anything")
        assert results == []
