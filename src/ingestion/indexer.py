"""Indexer — embeds and indexes enriched chunks into the vector store."""

import structlog

from src.ingestion.chunker import Chunk
from src.retrieval.vector_store import VectorStore

logger = structlog.get_logger()


class Indexer:
    """Embeds chunks and upserts them into the vector store."""

    def __init__(self):
        self._vector_store = VectorStore()

    async def index(self, chunks: list[Chunk]) -> int:
        """Embed and index a list of chunks. Expects enrichment already done.

        Returns:
            Number of chunks indexed.
        """
        await self._vector_store.ensure_collection()

        for chunk in chunks:
            await self._vector_store.upsert(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                metadata={
                    "source": chunk.source,
                    "grade": chunk.grade,
                    "subject": chunk.subject,
                    "unit": chunk.unit,
                    "chapter_number": chunk.chapter_number,
                    "chapter_title": chunk.chapter_title,
                    "section_number": chunk.section_number,
                    "section_title": chunk.section_title,
                    "subsection_number": chunk.subsection_number,
                    "subsection_title": chunk.subsection_title,
                    "page_number": chunk.page_number,
                    **chunk.metadata,
                },
                keywords=chunk.keywords,
                concepts=chunk.concepts,
            )

        logger.info("indexer.done", chunks_indexed=len(chunks))
        return len(chunks)
