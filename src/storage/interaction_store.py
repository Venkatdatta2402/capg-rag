"""Elasticsearch interaction store.

One ES document per session (keyed by session_id). Each interaction is appended
to the interactions array. Quiz submission updates the matching interaction
in-place using a Painless script.

In-memory fallback is used when ES is unavailable (dev / unit tests).

Requires: elasticsearch[async] >= 8.0
"""

import structlog
from datetime import datetime

from src.models.interaction import Interaction, SessionInteractionDocument, StudentResponse

logger = structlog.get_logger()

ES_INDEX = "session_interactions"

# In-memory fallback: session_id → SessionInteractionDocument
_store: dict[str, SessionInteractionDocument] = {}


def _get_es_client():
    """Lazily import and create the ES async client. Returns None if unavailable."""
    try:
        from elasticsearch import AsyncElasticsearch
        from config.settings import settings
        if not settings.elasticsearch_url:
            return None
        return AsyncElasticsearch(settings.elasticsearch_url)
    except ImportError:
        return None


class InteractionStore:
    """Append-only session interaction log backed by Elasticsearch."""

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    async def create_session(
        self,
        session_id: str,
        user_id: str,
        grade: str,
        first_interaction: Interaction,
    ) -> None:
        """Create the ES document for a new session with its first interaction."""
        doc = SessionInteractionDocument(
            session_id=session_id,
            user_id=user_id,
            grade=grade,
            interactions=[first_interaction],
        )
        _store[session_id] = doc

        es = _get_es_client()
        if not es:
            logger.debug("interaction_store.create.in_memory", session_id=session_id)
            return
        try:
            await es.index(
                index=ES_INDEX,
                id=session_id,
                document=doc.model_dump(mode="json"),
            )
            logger.info("interaction_store.created", session_id=session_id)
        except Exception as exc:
            logger.error("interaction_store.create.failed", session_id=session_id, error=str(exc))
        finally:
            await es.close()

    async def append_interaction(
        self,
        session_id: str,
        user_id: str,
        grade: str,
        interaction: Interaction,
    ) -> None:
        """Append a new interaction to an existing session document.

        Creates the session document if it doesn't exist yet (handles the
        edge case where the first create failed or session was lost).
        """
        # Update in-memory
        if session_id in _store:
            _store[session_id].interactions.append(interaction)
        else:
            _store[session_id] = SessionInteractionDocument(
                session_id=session_id,
                user_id=user_id,
                grade=grade,
                interactions=[interaction],
            )

        es = _get_es_client()
        if not es:
            logger.debug("interaction_store.append.in_memory", session_id=session_id)
            return
        try:
            script = {
                "script": {
                    "source": "ctx._source.interactions.add(params.interaction)",
                    "params": {"interaction": interaction.model_dump(mode="json")},
                },
                "upsert": SessionInteractionDocument(
                    session_id=session_id,
                    user_id=user_id,
                    grade=grade,
                    interactions=[interaction],
                ).model_dump(mode="json"),
            }
            await es.update(index=ES_INDEX, id=session_id, body=script)
            logger.info(
                "interaction_store.appended",
                session_id=session_id,
                interaction_id=interaction.interaction_id,
            )
        except Exception as exc:
            logger.error("interaction_store.append.failed", session_id=session_id, error=str(exc))
        finally:
            await es.close()

    # ------------------------------------------------------------------
    # Quiz submission update
    # ------------------------------------------------------------------

    async def update_quiz_response(
        self,
        session_id: str,
        interaction_id: str,
        student_response: StudentResponse,
        quiz_status: str = "submitted",
    ) -> None:
        """Update the student_response and quiz.status for a specific interaction."""
        # Update in-memory
        doc = _store.get(session_id)
        if doc:
            for ix in doc.interactions:
                if ix.interaction_id == interaction_id:
                    ix.student_response = student_response
                    ix.quiz.status = quiz_status
                    break

        es = _get_es_client()
        if not es:
            logger.debug("interaction_store.update_quiz.in_memory", session_id=session_id)
            return
        try:
            script = {
                "script": {
                    "source": """
                        for (int i = 0; i < ctx._source.interactions.length; i++) {
                            if (ctx._source.interactions[i].interaction_id == params.interaction_id) {
                                ctx._source.interactions[i].student_response = params.student_response;
                                ctx._source.interactions[i].quiz.status = params.quiz_status;
                            }
                        }
                    """,
                    "params": {
                        "interaction_id": interaction_id,
                        "student_response": student_response.model_dump(mode="json"),
                        "quiz_status": quiz_status,
                    },
                }
            }
            await es.update(index=ES_INDEX, id=session_id, body=script)
            logger.info(
                "interaction_store.quiz_updated",
                session_id=session_id,
                interaction_id=interaction_id,
                status=quiz_status,
            )
        except Exception as exc:
            logger.error("interaction_store.update_quiz.failed", session_id=session_id, error=str(exc))
        finally:
            await es.close()

    # ------------------------------------------------------------------
    # Read (for session review / analytics)
    # ------------------------------------------------------------------

    async def get_session(self, session_id: str) -> SessionInteractionDocument | None:
        """Fetch the full session document."""
        doc = _store.get(session_id)
        if doc:
            return doc

        es = _get_es_client()
        if not es:
            return None
        try:
            result = await es.get(index=ES_INDEX, id=session_id)
            return SessionInteractionDocument(**result["_source"])
        except Exception:
            return None
        finally:
            await es.close()
