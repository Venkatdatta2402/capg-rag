"""Context Object Builder.

Deterministic (no LLM) — assembles the context object from the learner profile
and session state. Used by the prompt selector to pick the right teaching style
and by the RAG agent to tailor generation.
"""

import structlog

from src.models.learner import LearnerProfile, SessionState
from src.models.query import ContextObject

logger = structlog.get_logger()


class ContextObjectBuilder:
    """Builds the context object from learner profile and session state.

    Not an LLM agent — pure deterministic logic.
    """

    def build(self, profile: LearnerProfile, session: SessionState) -> ContextObject:
        context = ContextObject(
            learner_grade=profile.grade,
            learning_styles=profile.learning_styles,
            technically_strong_areas=profile.technically_strong_areas,
            technically_weak_areas=profile.technically_weak_areas,
            softskills_strong_areas=profile.softskills_strong_areas,
            softskills_weak_areas=profile.softskills_weak_areas,
            retry_mode=session.retry_count > 0,
            retry_count=session.retry_count,
        )

        logger.info(
            "context_builder.built",
            grade=context.learner_grade,
            retry_mode=context.retry_mode,
        )
        return context
