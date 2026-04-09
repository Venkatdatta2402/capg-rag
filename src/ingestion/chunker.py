"""Structure-aware text chunking for NCERT knowledge base ingestion.

Chunking priority:
  1. Subsection boundaries (e.g. "5.1.1 Root Hair") — hardest split
  2. Section boundaries (e.g. "5.1 The Root")       — hard split
  3. Paragraph breaks (\n\n or blank lines)           — soft split
  4. Word-count limit (450 words) with 67-word overlap — fallback

Chunks never cross section or subsection boundaries.
Metadata (section_number, section_title, etc.) is inherited from the
boundary heading that opened the chunk.
"""

import re
import uuid

import structlog
from pydantic import BaseModel, Field

from src.ingestion.pdf_parser import ParsedPage

logger = structlog.get_logger()

CHUNK_SIZE = 450   # max words per chunk
OVERLAP    = 67    # words carried over from previous chunk on hard word-limit splits

# Matches NCERT subsection headings inline within page text (same patterns as parser)
_SUBSECTION_RE = re.compile(r"(\d{1,2}\.\d{1,2}\.\d{1,2})\s+([A-Z][^\n]{2,60})")
_SECTION_RE    = re.compile(r"(\d{1,2}\.\d{1,2})\s+([A-Z][^\n]{2,60})")


class Chunk(BaseModel):
    chunk_id: str
    text: str
    source: str = ""

    grade: str = ""
    subject: str = ""
    unit: str = ""
    chapter_number: str = ""
    chapter_title: str = ""
    section_number: str = ""
    section_title: str = ""
    subsection_number: str = ""
    subsection_title: str = ""

    page_number: int = 0
    keywords: list[str] = Field(default_factory=list)   # set by ChunkEnricher
    concepts: list[str] = Field(default_factory=list)   # set by ChunkEnricher
    metadata: dict = Field(default_factory=dict)


class Chunker:
    """Splits parsed pages into structure-aware chunks."""

    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP):
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
        chunks: list[Chunk] = []

        for page in pages:
            text = page.text.strip()
            if not text:
                continue

            # Base structural metadata from the parser (carried forward across pages)
            base_meta = page.metadata

            # Split page text on section/subsection headings found inline
            segments = _split_on_headings(text, base_meta)

            for seg_text, seg_meta in segments:
                seg_text = seg_text.strip()
                if not seg_text:
                    continue

                # Split each segment into paragraph-respecting word-limited chunks
                para_chunks = _split_into_chunks(seg_text, self._chunk_size, self._overlap)

                for chunk_text in para_chunks:
                    chunk_text = chunk_text.strip()
                    if not chunk_text:
                        continue
                    chunks.append(Chunk(
                        chunk_id=f"{source}_p{page.page_number}_{uuid.uuid4().hex[:6]}",
                        text=chunk_text,
                        source=source,
                        grade=grade,
                        subject=subject,
                        unit=unit,
                        chapter_number=chapter_number,
                        chapter_title=chapter_title,
                        section_number=seg_meta.get("section_number", ""),
                        section_title=seg_meta.get("section_title", ""),
                        subsection_number=seg_meta.get("subsection_number", ""),
                        subsection_title=seg_meta.get("subsection_title", ""),
                        page_number=page.page_number,
                        metadata={**base_meta, **seg_meta},
                    ))

        logger.info(
            "chunker.done",
            source=source,
            grade=grade,
            subject=subject,
            chunks=len(chunks),
        )
        return chunks


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _split_on_headings(text: str, base_meta: dict) -> list[tuple[str, dict]]:
    """Split page text at every section/subsection heading found inline.

    Returns list of (segment_text, metadata_dict) pairs. Metadata for each
    segment reflects the heading that opened it. Segments before any heading
    inherit base_meta from the parser.
    """
    # Find all heading positions (subsection first — more specific)
    boundaries: list[tuple[int, dict]] = []

    for m in _SUBSECTION_RE.finditer(text):
        num, title = m.group(1), m.group(2).strip()
        parts = num.split(".")
        sec_num = f"{parts[0]}.{parts[1]}"
        boundaries.append((m.start(), {
            "section_number": sec_num,
            "section_title": "",        # section title not re-derived here; carry from base
            "subsection_number": num,
            "subsection_title": title,
        }))

    for m in _SECTION_RE.finditer(text):
        num, title = m.group(1), m.group(2).strip()
        # Skip if this position is already covered by a subsection boundary
        if any(abs(b[0] - m.start()) < 5 for b in boundaries):
            continue
        boundaries.append((m.start(), {
            "section_number": num,
            "section_title": title,
            "subsection_number": "",
            "subsection_title": "",
        }))

    if not boundaries:
        return [(text, base_meta)]

    boundaries.sort(key=lambda x: x[0])

    segments: list[tuple[str, dict]] = []
    prev_pos = 0
    prev_meta = base_meta

    for pos, meta in boundaries:
        seg = text[prev_pos:pos]
        if seg.strip():
            segments.append((seg, prev_meta))
        prev_pos = pos
        prev_meta = {**base_meta, **meta}

    # Tail segment after last heading
    tail = text[prev_pos:]
    if tail.strip():
        segments.append((tail, prev_meta))

    return segments


def _split_into_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text respecting paragraph breaks, falling back to word-count limit.

    Strategy:
      - Split on paragraph breaks first (\n\n or lines with only whitespace).
      - Accumulate paragraphs until chunk_size would be exceeded.
      - When a single paragraph exceeds chunk_size, split it by words with overlap.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buffer: list[str] = []
    buffer_words = 0

    for para in paragraphs:
        para_words = len(para.split())

        if para_words > chunk_size:
            # Flush buffer first
            if buffer:
                chunks.append(" ".join(buffer))
                buffer, buffer_words = [], 0
            # Split oversized paragraph by words with overlap
            chunks.extend(_split_by_words(para, chunk_size, overlap))
            continue

        if buffer_words + para_words > chunk_size and buffer:
            chunks.append(" ".join(buffer))
            # Carry overlap words into next buffer
            overlap_text = " ".join(" ".join(buffer).split()[-overlap:])
            buffer = [overlap_text] if overlap_text else []
            buffer_words = len(overlap_text.split()) if overlap_text else 0

        buffer.append(para)
        buffer_words += para_words

    if buffer:
        chunks.append(" ".join(buffer))

    return chunks


def _split_by_words(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Hard word-count split with overlap for text that has no paragraph breaks."""
    words = text.split()
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start = end - overlap if end < len(words) else len(words)
    return chunks
