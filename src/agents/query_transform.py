"""Query Transformation Agent.

Focused purely on retrieval optimization — rewrites the raw student query
into a structured, keyword-enriched form that maximises NCERT chunk recall.
Domain keywords from the keyword store are injected so the model selects
precise NCERT terms rather than guessing.
"""

import structlog

from src.agents.base import BaseAgent
from src.llm.base import BaseLLMClient
from src.models.learner import LearnerProfile
from src.models.query import EnrichedQuery, QueryInput

logger = structlog.get_logger()

QUERY_TRANSFORM_PROMPT = """\
You are a query transformation agent for a CBSE/NCERT educational knowledge base.

Your ONLY job is to rewrite the student's raw query to maximize retrieval quality.

Given:
- Student grade: {grade}
- Subject context (if detectable): based on query content
- Raw query: {query}

Domain keyword pool (curated from the {grade} {subject} NCERT textbook):
{domain_keywords}

You must:
1. Identify the subject, topic, and sub-topic.
2. Select the most relevant keywords from the domain keyword pool above
   AND add any additional CBSE/NCERT terms you identify from the query.
3. Rewrite the query to be specific, unambiguous, and retrieval-friendly.
4. Include chapter and section hints if identifiable.

Respond in this exact format:
SUBJECT: <subject>
TOPIC: <topic>
SUB_TOPIC: <sub_topic>
CHAPTER_HINT: <chapter name if identifiable, else UNKNOWN>
SECTION_HINT: <section name if identifiable, else UNKNOWN>
QUERY_TYPE: <conceptual|procedural|conceptual_and_procedural>
KEYWORDS: <comma-separated list combining domain keywords + any additional terms>
REWRITTEN_QUERY: <the optimized query for retrieval>
"""


class QueryTransformAgent(BaseAgent):
    """Rewrites the raw query into a retrieval-optimized form with NCERT keywords."""

    def __init__(self, llm: BaseLLMClient):
        super().__init__(llm)

    async def run(
        self,
        query_input: QueryInput,
        profile: LearnerProfile,
        domain_keywords: list[str] | None = None,
    ) -> EnrichedQuery:
        """Transform the raw query for retrieval.

        Args:
            query_input: Raw user input.
            profile: Learner profile for grade context.
            domain_keywords: Curated keywords from the keyword store for this
                             grade+subject. Injected so the model selects precise
                             NCERT terms rather than guessing.
        """
        keywords_text = (
            ", ".join(domain_keywords) if domain_keywords
            else "No domain keywords available — infer from query content."
        )

        subject_hint = profile.grade  # model detects subject from query

        prompt = QUERY_TRANSFORM_PROMPT.format(
            grade=profile.grade,
            subject=subject_hint,
            query=query_input.query_text,
            domain_keywords=keywords_text,
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
