"""Adaptive retry loop logic shared across both architectures.

When a learner fails to demonstrate understanding, the system retries
with a narrower query and remedial prompt. Both architectures use the
same retry contract but differ in how they trigger it.
"""

import structlog

from config.settings import settings
from src.models.feedback import JudgeVerdict
from src.storage.session_memory import SessionMemoryStore
from src.storage.user_profile import UserProfileStore

logger = structlog.get_logger()


class RetryManager:
    """Manages the adaptive retry loop."""

    def __init__(
        self,
        session_store: SessionMemoryStore,
        profile_store: UserProfileStore,
        max_retries: int | None = None,
    ):
        self._sessions = session_store
        self._profiles = profile_store
        self._max_retries = max_retries or settings.max_retries

    async def should_retry(
        self, session_id: str, user_id: str, verdict: JudgeVerdict
    ) -> bool:
        """Determine if a retry should be attempted.

        In Architecture A: checks retry_mode flag.
        In Architecture B: uses the routing signal (verdict) directly.
        Both converge here for the retry count check.
        """
        if verdict.verdict == "UNDERSTOOD":
            return False

        session = await self._sessions.get(session_id, user_id)
        return session.retry_count < self._max_retries

    async def prepare_retry(
        self, session_id: str, user_id: str, failed_concept: str
    ) -> int:
        """Increment retry count and flag the knowledge gap.

        Returns:
            The new retry count.
        """
        retry_count = await self._sessions.increment_retry(session_id, user_id)
        await self._sessions.mark_knowledge_gap(session_id, user_id, failed_concept)

        logger.info(
            "retry.prepared",
            session_id=session_id,
            retry_count=retry_count,
            failed_concept=failed_concept,
        )
        return retry_count

    async def handle_max_retries(
        self, session_id: str, user_id: str, concept: str
    ) -> str:
        """Handle the case when max retries are exhausted.

        Flags the concept as a persistent weak area and returns a
        graceful escalation message.
        """
        await self._profiles.add_weak_area(user_id, concept)
        logger.warning(
            "retry.max_reached",
            user_id=user_id,
            concept=concept,
        )
        return (
            "This is a tricky concept. Let's come back to it later. "
            "Try asking your teacher or a parent to walk through it with you."
        )
