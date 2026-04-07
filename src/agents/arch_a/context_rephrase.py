"""Architecture A: Combined Context & Rephrase Agent.

In Arch A, a single agent handles both:
- Fetching learner profile + session state
- Parsing intent, rephrasing the query for retrieval
- Building the context object for prompt selection

This agent runs IN PARALLEL with raw query retrieval.
"""

import structlog

from src.agents.base import BaseAgent
from src.llm.base import BaseLLMClient
from src.models.learner import LearnerProfile, SessionState
from src.models.query import ContextObject, EnrichedQuery, QueryInput
from src.storage.session_memory import SessionMemoryStore
from src.storage.user_profile import UserProfileStore

logger = structlog.get_logger()

CONTEXT_REPHRASE_PROMPT = """\
You are a context enrichment agent for an educational RAG system.

Given the learner profile and their raw query, you must:
1. Identify the subject, topic, and sub-topic.
2. Determine the query type (conceptual, procedural, or both).
3. Rewrite the query to maximize retrieval precision against a CBSE/NCERT knowledge base.
4. Select the most relevant keywords from the domain keyword pool below,
   then add any additional NCERT-specific terms you identify.

Learner Profile:
- Grade: {grade}
- Board: {board}
- Comprehension: {comprehension_score}/10
- Learning style: {learning_style}
- Weak areas: {weak_areas}
- Mastered concepts: {mastered_concepts}

Session so far:
- Topics discussed: {topics_discussed}
- Concepts understood: {concepts_understood}
- Knowledge gaps: {knowledge_gaps}

Domain keyword pool (curated from the {grade} NCERT textbooks):
{domain_keywords}

Raw Query: {query}

Respond in this exact format:
SUBJECT: <subject>
TOPIC: <topic>
SUB_TOPIC: <sub_topic>
QUERY_TYPE: <conceptual|procedural|conceptual_and_procedural>
REWRITTEN_QUERY: <the enriched, retrieval-optimized query>
KEYWORDS: <comma-separated list combining domain keywords + any additional terms>
"""


class ContextRephraseAgent(BaseAgent):
    """Combined context enrichment and query rephrasing (Architecture A)."""

    def __init__(
        self,
        llm: BaseLLMClient,
        profile_store: UserProfileStore,
        session_store: SessionMemoryStore,
    ):
        super().__init__(llm)
        self._profiles = profile_store
        self._sessions = session_store

    async def run(
        self,
        query_input: QueryInput,
        domain_keywords: list[str] | None = None,
    ) -> tuple[EnrichedQuery, ContextObject]:
        """Enrich the query and build the context object in one pass.

        Args:
            query_input: Raw user input.
            domain_keywords: Curated keywords from the keyword store for this
                             learner's grade+subject, looked up before this agent runs.

        Returns:
            (EnrichedQuery, ContextObject) tuple.
        """
        profile = await self._profiles.get(query_input.user_id)
        session = await self._sessions.get(query_input.session_id, query_input.user_id)

        keywords_text = (
            ", ".join(domain_keywords) if domain_keywords
            else "No domain keywords available — infer from query content."
        )

        prompt = CONTEXT_REPHRASE_PROMPT.format(
            grade=profile.grade,
            board=profile.board,
            comprehension_score=profile.comprehension_score,
            learning_style=profile.learning_style,
            weak_areas=", ".join(profile.weak_areas) or "none",
            mastered_concepts=", ".join(profile.mastered_concepts) or "none",
            topics_discussed=", ".join(session.topics_discussed) or "none",
            concepts_understood=", ".join(session.concepts_understood) or "none",
            knowledge_gaps=", ".join(session.active_knowledge_gaps) or "none",
            domain_keywords=keywords_text,
            query=query_input.query_text,
        )

        response = await self._llm.generate(
            system_prompt="You are an educational context enrichment agent.",
            user_message=prompt,
        )

        enriched_query, context_obj = self._parse_response(
            response, query_input, profile, session
        )

        logger.info(
            "context_rephrase.done",
            original=query_input.query_text,
            rewritten=enriched_query.rewritten_text[:80],
        )
        return enriched_query, context_obj

    def _parse_response(
        self,
        response: str,
        query_input: QueryInput,
        profile: LearnerProfile,
        session: SessionState,
    ) -> tuple[EnrichedQuery, ContextObject]:
        """Parse the LLM response into structured objects."""
        lines = response.strip().split("\n")
        parsed = {}
        for line in lines:
            if ":" in line:
                key, _, value = line.partition(":")
                parsed[key.strip().upper()] = value.strip()

        enriched = EnrichedQuery(
            original_text=query_input.query_text,
            rewritten_text=parsed.get("REWRITTEN_QUERY", query_input.query_text),
            keywords=[k.strip() for k in parsed.get("KEYWORDS", "").split(",") if k.strip()],
            subject=parsed.get("SUBJECT", ""),
            topic=parsed.get("TOPIC", ""),
            sub_topic=parsed.get("SUB_TOPIC", ""),
            query_type=parsed.get("QUERY_TYPE", ""),
        )

        comp_level = "low" if profile.comprehension_score < 4 else (
            "low-medium" if profile.comprehension_score < 6 else (
                "medium" if profile.comprehension_score < 8 else "high"
            )
        )

        context = ContextObject(
            learner_grade=profile.grade,
            comprehension_level=comp_level,
            learning_style=profile.learning_style,
            weak_areas=profile.weak_areas,
            session_understood=session.concepts_understood,
            current_topic=enriched.topic,
            query_type=enriched.query_type,
            retry_mode=session.retry_count > 0,
            retry_count=session.retry_count,
        )

        return enriched, context
