"""Learner profile storage backed by PostgreSQL JSONB."""

import structlog

from src.models.learner import LearnerProfile

logger = structlog.get_logger()

# In-memory store for development; swap with PostgreSQL JSONB in production.
_profiles: dict[str, LearnerProfile] = {}


class UserProfileStore:
    """CRUD operations for learner profiles."""

    async def get(self, user_id: str) -> LearnerProfile:
        """Fetch a learner profile by user_id. Returns a default if not found."""
        if user_id in _profiles:
            return _profiles[user_id]
        logger.info("user_profile.not_found", user_id=user_id)
        return LearnerProfile(user_id=user_id)

    async def save(self, profile: LearnerProfile) -> None:
        """Persist a learner profile."""
        _profiles[profile.user_id] = profile
        logger.info("user_profile.saved", user_id=profile.user_id)

    async def add_mastered_concept(self, user_id: str, concept: str) -> None:
        """Add a concept to the learner's mastered list."""
        profile = await self.get(user_id)
        if concept not in profile.mastered_concepts:
            profile.mastered_concepts.append(concept)
            await self.save(profile)

    async def add_weak_area(self, user_id: str, area: str) -> None:
        """Flag a persistent weak area for the learner."""
        profile = await self.get(user_id)
        if area not in profile.weak_areas:
            profile.weak_areas.append(area)
            await self.save(profile)
