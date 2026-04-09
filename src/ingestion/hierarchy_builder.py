"""Hierarchy builder — generates one HierarchyNode per structural level per scope.

Groups chunks by (unit / chapter / section / subsection), concatenates their
text (truncated), calls the LLM to produce a 2-3 sentence summary, and returns
a flat list of HierarchyNodes ready for embedding + indexing.

Called once per ingestion run after chunks are produced.
"""

import structlog
from collections import defaultdict

from src.ingestion.chunker import Chunk
from src.llm.base import BaseLLMClient
from src.models.hierarchy import HierarchyNode

logger = structlog.get_logger()

_SUMMARY_SYSTEM_PROMPT = """\
You are summarising a section of a CBSE/NCERT textbook for use in a retrieval system.
Write a concise 2-3 sentence summary that captures:
- The main topic or concept covered
- Key ideas, processes, or terms introduced
- The educational level and subject context

Be specific and factual. Do not add interpretation beyond what the text says.
Respond with just the summary — no preamble.
"""

_MAX_CONTEXT_WORDS = 600   # truncate concatenated chunk text before sending to LLM
_SLUG_MAX = 40             # max chars for node_id slug


def _slug(text: str) -> str:
    return text.strip().replace(" ", "-")[:_SLUG_MAX]


def _truncate(text: str, max_words: int) -> str:
    words = text.split()
    return " ".join(words[:max_words]) + ("..." if len(words) > max_words else "")


class HierarchyBuilder:
    """Builds hierarchy nodes from a list of enriched chunks."""

    def __init__(self, llm: BaseLLMClient):
        self._llm = llm

    async def build(self, chunks: list[Chunk]) -> list[HierarchyNode]:
        """Return one HierarchyNode per unique level scope across all chunks."""
        nodes: list[HierarchyNode] = []
        nodes.extend(await self._build_level(chunks, "unit"))
        nodes.extend(await self._build_level(chunks, "chapter"))
        nodes.extend(await self._build_level(chunks, "section"))
        nodes.extend(await self._build_level(chunks, "subsection"))
        logger.info("hierarchy_builder.done", nodes=len(nodes))
        return nodes

    async def _build_level(
        self, chunks: list[Chunk], level: str
    ) -> list[HierarchyNode]:
        groups: dict[str, list[Chunk]] = defaultdict(list)

        for chunk in chunks:
            key = _group_key(chunk, level)
            if key:
                groups[key].append(chunk)

        nodes: list[HierarchyNode] = []
        for key, group in groups.items():
            node = await self._build_node(level, group)
            if node:
                nodes.append(node)

        logger.debug("hierarchy_builder.level_done", level=level, nodes=len(nodes))
        return nodes

    async def _build_node(
        self, level: str, chunks: list[Chunk]
    ) -> HierarchyNode | None:
        if not chunks:
            return None

        ref = chunks[0]
        combined_text = " ".join(c.text for c in chunks)
        truncated = _truncate(combined_text, _MAX_CONTEXT_WORDS)

        title, node_id = _node_title_and_id(level, ref)
        if not title:
            return None

        user_msg = (
            f"Grade: {ref.grade} | Subject: {ref.subject} | {level.capitalize()}: {title}\n\n"
            f"Content:\n{truncated}"
        )
        try:
            summary = await self._llm.generate(
                system_prompt=_SUMMARY_SYSTEM_PROMPT,
                user_message=user_msg,
            )
            summary = summary.strip()
        except Exception as exc:
            logger.warning(
                "hierarchy_builder.summary_failed",
                node_id=node_id, error=str(exc),
            )
            summary = truncated[:300]

        # Aggregate unique keywords and concepts across all chunks in this node
        agg_keywords: list[str] = list({kw for c in chunks for kw in c.keywords})
        agg_concepts: list[str] = list({ct for c in chunks for ct in c.concepts})

        return HierarchyNode(
            node_id=node_id,
            node_type=level,
            title=title,
            summary=summary,
            grade=ref.grade,
            subject=ref.subject,
            unit=ref.unit,
            chapter_title=ref.chapter_title,
            section_number=ref.section_number if level in ("section", "subsection") else "",
            section_title=ref.section_title if level in ("section", "subsection") else "",
            subsection_number=ref.subsection_number if level == "subsection" else "",
            subsection_title=ref.subsection_title if level == "subsection" else "",
            keywords=agg_keywords,
            concepts=agg_concepts,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _group_key(chunk: Chunk, level: str) -> str:
    """Return a unique grouping key for the chunk at the given level.
    Empty string means this chunk has no metadata for that level — skip it.
    """
    if level == "unit":
        return f"{chunk.grade}|{chunk.subject}|{chunk.unit}" if chunk.unit else ""
    if level == "chapter":
        return (
            f"{chunk.grade}|{chunk.subject}|{chunk.unit}|{chunk.chapter_title}"
            if chunk.chapter_title else ""
        )
    if level == "section":
        return (
            f"{chunk.grade}|{chunk.subject}|{chunk.chapter_title}|{chunk.section_number}"
            if chunk.section_number else ""
        )
    if level == "subsection":
        return (
            f"{chunk.grade}|{chunk.subject}|{chunk.section_number}|{chunk.subsection_number}"
            if chunk.subsection_number else ""
        )
    return ""


def _node_title_and_id(level: str, ref: Chunk) -> tuple[str, str]:
    prefix = f"{ref.grade}_{ref.subject}"
    if level == "unit":
        title = ref.unit
        return title, f"{prefix}_unit_{_slug(title)}"
    if level == "chapter":
        title = ref.chapter_title
        return title, f"{prefix}_chapter_{_slug(title)}"
    if level == "section":
        title = ref.section_title or ref.section_number
        return title, f"{prefix}_section_{_slug(ref.section_number)}"
    if level == "subsection":
        title = ref.subsection_title or ref.subsection_number
        return title, f"{prefix}_subsection_{_slug(ref.subsection_number)}"
    return "", ""
