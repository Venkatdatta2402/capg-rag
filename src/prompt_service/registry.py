"""Prompt Registry — versioned prompt storage and management."""

import structlog

from src.models.prompt import PromptCandidate, PromptVersion

logger = structlog.get_logger()

# In-memory registry; production uses PostgreSQL.
_versions: dict[str, PromptVersion] = {}
_candidates: dict[str, PromptCandidate] = {}


class PromptRegistry:
    """CRUD operations for versioned prompts and candidates."""

    async def register(self, version: PromptVersion) -> None:
        """Register a new prompt version."""
        _versions[version.version_id] = version
        logger.info("prompt_registry.registered", version_id=version.version_id)

    async def get(self, version_id: str) -> PromptVersion | None:
        """Fetch a prompt version by ID."""
        return _versions.get(version_id)

    async def list_active(self, grade: str = "", variant: str = "") -> list[PromptVersion]:
        """List active prompt versions, optionally filtered by grade and variant."""
        results = [v for v in _versions.values() if v.status == "active"]
        if grade:
            results = [v for v in results if v.grade == grade]
        if variant:
            results = [v for v in results if v.variant == variant]
        return results

    async def list_candidates(self) -> list[PromptCandidate]:
        """List all prompt candidates currently in testing."""
        return [c for c in _candidates.values() if c.status == "testing"]

    async def add_candidate(self, candidate: PromptCandidate) -> None:
        """Add a new prompt candidate for canary testing."""
        _candidates[candidate.candidate_id] = candidate
        logger.info("prompt_registry.candidate_added", candidate_id=candidate.candidate_id)

    async def promote_candidate(self, candidate_id: str) -> None:
        """Promote a candidate to active status, retiring the parent version."""
        candidate = _candidates.get(candidate_id)
        if not candidate:
            return
        candidate.status = "promoted"
        # Retire the parent
        parent = _versions.get(candidate.parent_version_id)
        if parent:
            parent.status = "retired"
        # Create new active version from candidate
        new_version = PromptVersion(
            version_id=candidate.candidate_id,
            template=candidate.template,
            grade=parent.grade if parent else "",
            variant=parent.variant if parent else "standard",
            status="active",
        )
        _versions[new_version.version_id] = new_version
        logger.info("prompt_registry.promoted", candidate_id=candidate_id)

    async def reject_candidate(self, candidate_id: str) -> None:
        """Reject a candidate — it will not be deployed."""
        candidate = _candidates.get(candidate_id)
        if candidate:
            candidate.status = "rejected"
