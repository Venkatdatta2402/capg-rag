"""Retrieval result models."""

from pydantic import BaseModel, Field


class RetrievalResult(BaseModel):
    """A single chunk retrieved from the knowledge base.

    Metadata mirrors the NCERT book structure:
      grade + subject → unit/part → chapter → section (5.1) → subsection (5.1.1)
    """

    chunk_id: str
    text: str
    source: str = ""                # e.g. "NCERT Class 5 Maths Ch.6 pg.72"

    # Scope (always present — set at ingestion)
    grade: str = ""                 # e.g. "5", "10"
    subject: str = ""               # e.g. "Maths", "Science", "Social Science"

    # NCERT hierarchy (set at ingestion where detectable)
    unit: str = ""                  # e.g. "Unit II", "Part I" (groups chapters)
    chapter_number: str = ""        # e.g. "6"
    chapter_title: str = ""         # e.g. "Measurement"
    section_number: str = ""        # e.g. "5.1" (major heading)
    section_title: str = ""         # e.g. "The Root"
    subsection_number: str = ""     # e.g. "5.1.1" (minor heading)
    subsection_title: str = ""      # e.g. "Root Hair"

    score: float = 0.0              # similarity or BM25 score
    keywords: list[str] = Field(default_factory=list)   # chunk-level domain keywords (from ingestion enrichment)
    concepts: list[str] = Field(default_factory=list)   # chunk-level high-level concepts (from ingestion enrichment)
    metadata: dict = Field(default_factory=dict)


class RerankedChunk(BaseModel):
    """A chunk after cross-encoder reranking."""

    chunk_id: str
    text: str
    source: str = ""

    # Carry NCERT hierarchy through for citation and filtering
    grade: str = ""
    subject: str = ""
    unit: str = ""
    chapter_number: str = ""
    chapter_title: str = ""
    section_number: str = ""
    section_title: str = ""
    subsection_number: str = ""
    subsection_title: str = ""

    original_score: float = 0.0
    rerank_score: float = 0.0
    keywords: list[str] = Field(default_factory=list)
    concepts: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
