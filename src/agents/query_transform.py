"""Query Transformation Agent.

Rewrites the raw student query into a retrieval-optimized form.
Receives chunk-level keywords and concepts from the initial broad retrieval pass
so it can incorporate precise NCERT signals into the rewrite rather than guessing.

session_context is passed for normal queries (disambiguation from prior session topics).
It is omitted when the question comes from the judge (already focused).
"""

import structlog

from src.agents.base import BaseAgent
from src.llm.base import BaseLLMClient
from src.models.query import EnrichedQuery, QueryInput
from src.models.retrieval import RetrievalResult

logger = structlog.get_logger()

QUERY_TRANSFORM_PROMPT = """\
You are a query transformation agent for a CBSE/NCERT educational knowledge base.

Your job is to rewrite the student's raw query to maximize second-pass retrieval quality.

Given:
- Student grade: {grade}
- Raw query: {query}
{session_block}\
Signals extracted from initially retrieved chunks (keywords and concepts from the knowledge base):
Keywords: {chunk_keywords}
Concepts: {chunk_concepts}

You must:
1. Identify the subject, topic, and sub-topic.
2. From the chunk signals above, select the keywords and concepts that are directly \
relevant to the query. Discard unrelated ones. Add any additional NCERT terms you identify.
3. Rewrite the query to be specific, unambiguous, and retrieval-friendly, incorporating \
the selected signals.
4. Include chapter and section hints if identifiable.

Respond in this exact format:
SUBJECT: <subject>
TOPIC: <topic>
SUB_TOPIC: <sub_topic>
CHAPTER_HINT: <chapter name if identifiable, else UNKNOWN>
SECTION_HINT: <section name if identifiable, else UNKNOWN>
QUERY_TYPE: <conceptual|procedural|conceptual_and_procedural>
KEYWORDS: <comma-separated list of selected + additional NCERT terms>
REWRITTEN_QUERY: <the optimized query for retrieval>
"""


class QueryTransformAgent(BaseAgent):
    """Rewrites the raw query using chunk signals from initial retrieval."""

    def __init__(self, llm: BaseLLMClient):
        super().__init__(llm)

    async def run(
        self,
        query_input: QueryInput,
        grade: str,
        initial_chunks: list[RetrievalResult] | None = None,
        session_context: str = "",
    ) -> EnrichedQuery:
        """Transform the raw query for second-pass retrieval.

        Args:
            query_input: Raw user input.
            grade: Learner grade for NCERT scope.
            initial_chunks: Chunks from initial broad retrieval — keywords/concepts
                            are pooled and passed as signals for the rewrite.
            session_context: Session history summary for query disambiguation.
                             Empty when query comes from the judge.
        """
        chunk_keywords, chunk_concepts = _pool_signals(initial_chunks or [])

        session_block = (
            f"Session context:\n{session_context}\n\n"
            if session_context else ""
        )

        prompt = QUERY_TRANSFORM_PROMPT.format(
            grade=grade,
            query=query_input.query_text,
            session_block=session_block,
            chunk_keywords=", ".join(chunk_keywords) if chunk_keywords else "none",
            chunk_concepts=", ".join(chunk_concepts) if chunk_concepts else "none",
        )

        response = await self._llm.generate(
            system_prompt="You are a retrieval query optimization agent.",
            user_message=prompt,
        )

        enriched = self._parse_response(response, query_input)
        logger.info(
            "query_transform.done",
            original=query_input.query_text,
            rewritten=enriched.rewritten_text[:80],
            keywords=enriched.keywords,
        )
        return enriched

    def _parse_response(self, response: str, query_input: QueryInput) -> EnrichedQuery:
        parsed = {}
        for line in response.strip().split("\n"):
            if ":" in line:
                key, _, value = line.partition(":")
                parsed[key.strip().upper()] = value.strip()

        return EnrichedQuery(
            original_text=query_input.query_text,
            rewritten_text=parsed.get("REWRITTEN_QUERY", query_input.query_text),
            keywords=[k.strip() for k in parsed.get("KEYWORDS", "").split(",") if k.strip()],
            subject=parsed.get("SUBJECT", ""),
            topic=parsed.get("TOPIC", ""),
            sub_topic=parsed.get("SUB_TOPIC", ""),
            query_type=parsed.get("QUERY_TYPE", ""),
        )


def _pool_signals(chunks: list[RetrievalResult]) -> tuple[list[str], list[str]]:
    """Deduplicate and pool keywords + concepts across all initial chunks."""
    seen_kw: set[str] = set()
    seen_co: set[str] = set()
    keywords: list[str] = []
    concepts: list[str] = []

    for chunk in chunks:
        for kw in chunk.keywords:
            if kw.lower() not in seen_kw:
                seen_kw.add(kw.lower())
                keywords.append(kw)
        for co in chunk.concepts:
            if co.lower() not in seen_co:
                seen_co.add(co.lower())
                concepts.append(co)

    return keywords, concepts
