"""Text chunking strategies for knowledge base ingestion."""

import uuid

import structlog
from pydantic import BaseModel, Field

from src.ingestion.pdf_parser import ParsedPage

logger = structlog.get_logger()


class Chunk(BaseModel):
    """A chunk of text ready for embedding and indexing.

    Metadata mirrors the NCERT book structure so the hierarchical
    retriever can progressively narrow from broad to precise scope:
      grade + subject → unit/part → chapter → section → subsection
    """

    chunk_id: str
    text: str
    source: str = ""                # e.g. "NCERT_Grade5_Maths_Ch6_pg72"

    # Scope — supplied at ingestion time, not derived from text
    grade: str = ""                 # e.g. "5", "10"
    subject: str = ""               # e.g. "Maths", "Science", "Social Science"

    # NCERT hierarchy — extracted from PDF structure during parsing
    unit: str = ""                  # e.g. "Unit II", "Part I"
    chapter_number: str = ""        # e.g. "6"
    chapter_title: str = ""         # e.g. "Measurement"
    section_number: str = ""        # e.g. "5.1"
    section_title: str = ""         # e.g. "The Root"
    subsection_number: str = ""     # e.g. "5.1.1"
    subsection_title: str = ""      # e.g. "Root Hair"

    page_number: int = 0
    metadata: dict = Field(default_factory=dict)


class Chunker:
    """Splits parsed pages into chunks for vector indexing.

    Supports configurable chunk size and overlap.
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self._chunk_size = chunk_size
        self._overlap = overlap

    def chunk(
        self,
        pages: list[ParsedPage],
        source: str = "",
        grade: str = "",
        subject: str = "",
        unit: str = "",
        chapter_number: str = "",
        chapter_title: str = "",
    ) -> list[Chunk]:
        """Split pages into overlapping text chunks.

        Args:
            pages: Parsed PDF pages.
            source: Source document identifier.
            grade: Learner grade (e.g. "5", "10") — supplied by the caller.
            subject: Subject name (e.g. "Maths") — supplied by the caller.
            unit: Unit/Part grouping (e.g. "Unit II") — supplied by the caller if known.
            chapter_number: Chapter number string — supplied by the caller.
            chapter_title: Chapter title — supplied by the caller.

        Note:
            Section and subsection metadata are extracted per-page from
            page.metadata (populated by the PDF parser when it detects
            decimal-numbered headings like "5.1" or "5.1.1").
        """
        chunks = []

        for page in pages:
            text = page.text.strip()
            if not text:
                continue

            # Section/subsection extracted by the PDF parser per page
            section_number = page.metadata.get("section_number", "")
            section_title = page.metadata.get("section_title", "")
            subsection_number = page.metadata.get("subsection_number", "")
            subsection_title = page.metadata.get("subsection_title", "")

            words = text.split()
            start = 0
            while start < len(words):
                end = start + self._chunk_size
                chunk_text = " ".join(words[start:end])

                chunks.append(Chunk(
                    chunk_id=f"{source}_p{page.page_number}_{uuid.uuid4().hex[:6]}",
                    text=chunk_text,
                    source=source,
                    grade=grade,
                    subject=subject,
                    unit=unit,
                    chapter_number=chapter_number,
                    chapter_title=chapter_title,
                    section_number=section_number,
                    section_title=section_title,
                    subsection_number=subsection_number,
                    subsection_title=subsection_title,
                    page_number=page.page_number,
                    metadata=page.metadata,
                ))

                start = end - self._overlap if end < len(words) else len(words)

        logger.info("chunker.done", source=source, grade=grade, subject=subject, chunks=len(chunks))
        return chunks
