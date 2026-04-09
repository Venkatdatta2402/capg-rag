"""Chunk enrichment at ingestion time.

For each chunk, generates:
  keywords  — 5-8 domain-specific, non-redundant NCERT terms present in the chunk
  concepts  — 2-4 high-level ideas the chunk is about

True batching: all chunks in a batch are sent in ONE LLM call.
The LLM returns one KEYWORDS/CONCEPTS block per chunk, indexed by chunk number.

Usage:
    enricher = ChunkEnricher(llm)
    enriched_chunks = [await enricher.enrich(chunk) for chunk in chunks]
"""

import structlog

from src.ingestion.chunker import Chunk
from src.llm.base import BaseLLMClient

logger = structlog.get_logger()

_BATCH_SIZE = 8

_SYSTEM_PROMPT = """\
You are a curriculum analyst for CBSE/NCERT textbooks.
You will receive multiple numbered text chunks from an NCERT textbook.
For EACH chunk, extract:
  keywords: 5-8 domain-specific, non-redundant terms (subject-specific nouns, process names, \
defined vocabulary). No common English words or vague terms.
  concepts: 2-4 high-level ideas or principles the chunk is about (2-5 word phrases).

Respond with a JSON array — one object per chunk, in the same order, no extra text:
[
  {"keywords": ["term1", "term2", ...], "concepts": ["phrase1", "phrase2", ...]},
  {"keywords": [...], "concepts": [...]},
  ...
]
"""


class ChunkEnricher:
    """Enriches chunks with LLM-generated keywords and concepts using true batch calls."""

    def __init__(self, llm: BaseLLMClient):
        self._llm = llm

    async def enrich(self, chunk: Chunk) -> Chunk:
        """Enrich a single chunk. Delegates to batch internally."""
        results = await self._enrich_batch([chunk])
        return results[0]

    async def enrich_all(self, chunks: list[Chunk]) -> list[Chunk]:
        """Enrich all chunks using true batching (one LLM call per batch of 8)."""
        enriched: list[Chunk] = []
        batches = [chunks[i:i + _BATCH_SIZE] for i in range(0, len(chunks), _BATCH_SIZE)]
        for idx, batch in enumerate(batches):
            results = await self._enrich_batch(batch)
            enriched.extend(results)
            logger.debug("chunk_enricher.batch_done", batch=idx + 1, total=len(batches))
        logger.info("chunk_enricher.done", total=len(enriched))
        return enriched

    async def _enrich_batch(self, batch: list[Chunk]) -> list[Chunk]:
        """Send all chunks in one LLM call and parse per-chunk results."""
        user_msg = "\n\n".join(
            f"CHUNK_{i + 1} [Grade: {c.grade} | Subject: {c.subject} | Chapter: {c.chapter_title}]:\n{c.text[:800]}"
            for i, c in enumerate(batch)
        )

        try:
            response = await self._llm.generate(
                system_prompt=_SYSTEM_PROMPT,
                user_message=user_msg,
            )
            parsed = _parse_batch(response, len(batch))
        except Exception as exc:
            logger.warning("chunk_enricher.batch_failed", error=str(exc))
            parsed = [({}, {})] * len(batch)

        return [
            chunk.model_copy(update={"keywords": kw, "concepts": co})
            for chunk, (kw, co) in zip(batch, parsed)
        ]


def _parse_batch(response: str, expected: int) -> list[tuple[list[str], list[str]]]:
    """Parse JSON array response into per-chunk (keywords, concepts) tuples."""
    import json

    # Strip markdown code fences if present
    text = response.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        items = json.loads(text)
        if not isinstance(items, list):
            raise ValueError("expected a JSON array")
    except Exception as exc:
        logger.warning("chunk_enricher.parse_failed", error=str(exc))
        return [([], [])] * expected

    result = []
    for i in range(expected):
        if i < len(items) and isinstance(items[i], dict):
            kw = [str(k) for k in items[i].get("keywords", []) if k]
            co = [str(c) for c in items[i].get("concepts", []) if c]
        else:
            kw, co = [], []
        result.append((kw, co))
    return result
