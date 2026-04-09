"""Session memory backed by Elasticsearch with in-memory fallback.

One ES document per session (index: session_memory, keyed by session_id).
The in-memory dict is the primary read source; ES is updated on every write
so state survives restarts and is shared across workers.

TTL is managed via an ES ILM policy on the index (recommended: delete after 4h).

Structure differs from InteractionStore (session_interactions):
  - session_memory  → recent window + rolling summary + retry state
  - session_interactions → full interaction log with quiz data and context chunks
"""

import structlog

from src.models.learner import PastSummary, RecentInteraction, SessionState

logger = structlog.get_logger()

ES_INDEX = "session_memory"
_RECENT_WINDOW = 5   # keep last N interactions; older ones roll into summary

# In-memory primary store: session_id → SessionState
_sessions: dict[str, SessionState] = {}


# ---------------------------------------------------------------------------
# ES client (lazy)
# ---------------------------------------------------------------------------

def _get_es_client():
    """Lazily create the ES async client. Returns None if unavailable."""
    try:
        from elasticsearch import AsyncElasticsearch
        from config.settings import settings
        if not settings.elasticsearch_url:
            return None
        return AsyncElasticsearch(settings.elasticsearch_url)
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Deterministic summary helpers
# ---------------------------------------------------------------------------

def _compute_trend(scores: list[int]) -> str:
    if len(scores) < 4:
        return "unknown"
    mid = len(scores) // 2
    first_avg = sum(scores[:mid]) / mid
    second_avg = sum(scores[mid:]) / (len(scores) - mid)
    if second_avg > first_avg + 0.2:
        return "improving"
    if first_avg > second_avg + 0.2:
        return "declining"
    return "stable"


def _archive_interaction(summary: PastSummary, ix: RecentInteraction) -> None:
    """Roll a sliding interaction into the PastSummary deterministically."""
    if ix.topic and ix.topic not in summary.covered_topics:
        summary.covered_topics.append(ix.topic)

    if ix.quiz_status == "submitted":
        summary.archived_scores.append(ix.score)
        summary.performance_trend = _compute_trend(summary.archived_scores)

        if ix.score == 0:
            if ix.question not in summary.common_errors:
                summary.common_errors.append(ix.question)
            if ix.question not in summary.key_misconceptions:
                summary.key_misconceptions.append(ix.question)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class SessionMemoryStore:
    """Short-term session state: recent interactions + rolling summary."""

    # ----------------------------------------------------------------
    # Core read / write
    # ----------------------------------------------------------------

    async def get(self, session_id: str, user_id: str) -> SessionState:
        """Return session state; creates a fresh one if not found."""
        if session_id in _sessions:
            return _sessions[session_id]

        es = _get_es_client()
        if es:
            try:
                result = await es.get(index=ES_INDEX, id=session_id)
                state = SessionState(**result["_source"])
                _sessions[session_id] = state
                return state
            except Exception:
                pass
            finally:
                await es.close()

        return SessionState(session_id=session_id, user_id=user_id)

    async def save(self, state: SessionState) -> None:
        """Update in-memory store and upsert full document to ES."""
        _sessions[state.session_id] = state

        es = _get_es_client()
        if not es:
            logger.debug("session_memory.save.in_memory", session_id=state.session_id)
            return
        try:
            await es.index(
                index=ES_INDEX,
                id=state.session_id,
                document=state.model_dump(mode="json"),
            )
            logger.debug("session_memory.saved", session_id=state.session_id)
        except Exception as exc:
            logger.error("session_memory.save.failed", session_id=state.session_id, error=str(exc))
        finally:
            await es.close()

    # ----------------------------------------------------------------
    # Session mutations
    # ----------------------------------------------------------------

    async def append_interaction(
        self,
        session_id: str,
        user_id: str,
        interaction: RecentInteraction,
    ) -> None:
        """Add interaction to the recent window; archive oldest if window exceeds limit."""
        state = await self.get(session_id, user_id)
        state.recent_interactions.append(interaction)

        if len(state.recent_interactions) > _RECENT_WINDOW:
            oldest = state.recent_interactions.pop(0)
            _archive_interaction(state.summary_of_past, oldest)
            logger.debug(
                "session_memory.archived",
                session_id=session_id,
                topic=oldest.topic,
            )

        await self.save(state)

    async def save_context(
        self,
        session_id: str,
        user_id: str,
        context_obj,   # ContextObject — typed loosely to avoid circular import at module level
    ) -> None:
        """Persist the ContextObject on the session after first build."""
        state = await self.get(session_id, user_id)
        state.context_object = context_obj
        await self.save(state)

    async def set_retry_mode(self, session_id: str, user_id: str) -> None:
        """Called by quiz route on FAILED verdict.

        Increments retry_count and sets retry_mode=True on the persisted
        ContextObject so the next pipeline run picks it up without rebuilding.
        """
        state = await self.get(session_id, user_id)
        state.retry_count += 1
        if state.context_object is not None:
            state.context_object.retry_mode = True
            state.context_object.retry_count = state.retry_count
        await self.save(state)
        logger.info(
            "session_memory.retry_mode_set",
            session_id=session_id,
            retry_count=state.retry_count,
        )

    async def update_quiz_result(
        self,
        session_id: str,
        user_id: str,
        interaction_id: str,
        score: int,
        quiz_status: str,
    ) -> None:
        """Mirror quiz submission outcome into the recent interaction window."""
        state = await self.get(session_id, user_id)
        for ix in state.recent_interactions:
            if ix.interaction_id == interaction_id:
                ix.quiz_status = quiz_status
                ix.score = score
                break
        else:
            logger.warning(
                "session_memory.update_quiz.not_found",
                session_id=session_id,
                interaction_id=interaction_id,
            )
        await self.save(state)

    async def delete(self, session_id: str) -> None:
        """Remove session from in-memory store and ES after end-of-session review."""
        _sessions.pop(session_id, None)

        es = _get_es_client()
        if not es:
            return
        try:
            await es.delete(index=ES_INDEX, id=session_id, ignore=[404])
        except Exception as exc:
            logger.warning("session_memory.delete.failed", session_id=session_id, error=str(exc))
        finally:
            await es.close()
