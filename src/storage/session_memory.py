"""Session memory backed by Redis (in-memory fallback for development)."""

import json

import structlog

from src.models.learner import SessionState

logger = structlog.get_logger()

# In-memory fallback; production uses Redis with TTL-based expiry.
_sessions: dict[str, SessionState] = {}


class SessionMemoryStore:
    """Short-term session state management."""

    async def get(self, session_id: str, user_id: str) -> SessionState:
        """Fetch session state. Returns a fresh session if none exists."""
        if session_id in _sessions:
            return _sessions[session_id]
        return SessionState(session_id=session_id, user_id=user_id)

    async def save(self, state: SessionState) -> None:
        """Persist session state."""
        _sessions[state.session_id] = state
        logger.debug("session_memory.saved", session_id=state.session_id)

    async def mark_concept_understood(self, session_id: str, user_id: str, concept: str) -> None:
        """Mark a concept as understood in the current session."""
        state = await self.get(session_id, user_id)
        if concept not in state.concepts_understood:
            state.concepts_understood.append(concept)
        if concept in state.active_knowledge_gaps:
            state.active_knowledge_gaps.remove(concept)
        await self.save(state)

    async def mark_knowledge_gap(self, session_id: str, user_id: str, concept: str) -> None:
        """Flag an active knowledge gap in the current session."""
        state = await self.get(session_id, user_id)
        if concept not in state.active_knowledge_gaps:
            state.active_knowledge_gaps.append(concept)
        await self.save(state)

    async def increment_retry(self, session_id: str, user_id: str) -> int:
        """Increment and return the retry count for this session."""
        state = await self.get(session_id, user_id)
        state.retry_count += 1
        await self.save(state)
        return state.retry_count
