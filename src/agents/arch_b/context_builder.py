"""Architecture B: Context Object Builder.

Responsible only for constructing the context object used for prompt
selection and downstream orchestration. Separated from query transformation
for modularity.
"""

import structlog

from src.models.learner import LearnerProfile, SessionState
from src.models.query import ContextObject, EnrichedQuery

logger = structlog.get_logger()


class ContextObjectBuilder:
    """Builds the context object from learner profile and session state (Architecture B).

    This is NOT an LLM-powered agent — it's deterministic logic that assembles
    context from structured data sources.
    """

    def build(
        self,
        profile: LearnerProfile,
        session: SessionState,
        enriched_query: EnrichedQuery,
    ) -> ContextObject:
        """Build the context object for prompt selection.

        Args:
            profile: Long-term learner profile.
            session: Current session state.
            enriched_query: Output from the Query Transformation Agent.
        """
        comp_level = self._compute_comprehension_level(profile.comprehension_score)

        context = ContextObject(
            learner_grade=profile.grade,
            comprehension_level=comp_level,
            learning_style=profile.learning_style,
            weak_areas=profile.weak_areas,
            session_understood=session.concepts_understood,
            current_topic=enriched_query.topic,
            query_type=enriched_query.query_type,
            retry_mode=session.retry_count > 0,
            retry_count=session.retry_count,
        )

        logger.info(
            "context_builder.built",
            grade=context.learner_grade,
            comp_level=context.comprehension_level,
            topic=context.current_topic,
        )
        return context

    def _compute_comprehension_level(self, score: float) -> str:
        if score < 4:
            return "low"
        if score < 6:
            return "low-medium"
        if score < 8:
            return "medium"
        return "high"
