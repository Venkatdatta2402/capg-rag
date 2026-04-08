"""Session lifecycle endpoint.

POST /session/end
  Fire-and-forget: schedules an end-of-session review as a FastAPI BackgroundTask.
  The review agent reads the full Q&A history, updates the learner's long-term
  profile (technically_strong/weak, softskills, learning_style), writes a
  governance FeedbackRecord, then deletes the session.

  Returns immediately with {"status": "review_scheduled"} so the client is not
  blocked by the LLM review call.
"""

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends
from pydantic import BaseModel

from api.dependencies import (
    get_feedback_store,
    get_profile_store,
    get_session_store,
    get_session_review_agent,
)
from src.agents.session_review import SessionReviewAgent
from src.storage.feedback_store import FeedbackStore
from src.storage.session_memory import SessionMemoryStore
from src.storage.user_profile import UserProfileStore

router = APIRouter()
logger = structlog.get_logger()


class SessionEndRequest(BaseModel):
    session_id: str
    user_id: str


class SessionEndResponse(BaseModel):
    status: str


@router.post("/session/end", response_model=SessionEndResponse)
async def end_session(
    request: SessionEndRequest,
    background_tasks: BackgroundTasks,
    session_store: SessionMemoryStore = Depends(get_session_store),
    profile_store: UserProfileStore = Depends(get_profile_store),
    feedback_store: FeedbackStore = Depends(get_feedback_store),
    review_agent: SessionReviewAgent = Depends(get_session_review_agent),
):
    """Schedule an end-of-session review and return immediately.

    The background task:
      1. Loads session history and learner profile
      2. Runs SessionReviewAgent (LLM) to infer updated profile fields
      3. Persists updated LearnerProfile to long-term storage
      4. Writes a FeedbackRecord to the governance pipeline
      5. Deletes the session from Redis/in-memory store
    """
    background_tasks.add_task(
        _run_session_review,
        session_id=request.session_id,
        user_id=request.user_id,
        session_store=session_store,
        profile_store=profile_store,
        feedback_store=feedback_store,
        review_agent=review_agent,
    )
    logger.info("session.end_scheduled", session_id=request.session_id, user_id=request.user_id)
    return SessionEndResponse(status="review_scheduled")


async def _run_session_review(
    session_id: str,
    user_id: str,
    session_store: SessionMemoryStore,
    profile_store: UserProfileStore,
    feedback_store: FeedbackStore,
    review_agent: SessionReviewAgent,
) -> None:
    """Background task: run the session review and update long-term profile."""
    try:
        session = await session_store.get(session_id, user_id)
        profile = await profile_store.get(user_id)

        if not session.recent_interactions and not session.summary_of_past.covered_topics:
            logger.info("session.review_skipped.no_history", session_id=session_id)
            await session_store.delete(session_id)
            return

        result = await review_agent.review(profile, session)

        # Apply reviewed fields to long-term profile
        profile.technically_strong_areas = result.technically_strong
        profile.technically_weak_areas = result.technically_weak
        profile.softskills_strong_areas = result.softskills_strong
        profile.softskills_weak_areas = result.softskills_weak
        if result.learning_styles:
            profile.learning_styles = result.learning_styles

        await profile_store.save(profile)

        # Send governance feedback so the prompt improvement loop can act on it
        feedback_record = review_agent.build_feedback_record(result, session, grade=profile.grade)
        await feedback_store.write(feedback_record)

        await session_store.delete(session_id)

        logger.info(
            "session.review_complete",
            session_id=session_id,
            user_id=user_id,
            learning_style=result.learning_style,
        )

    except Exception as exc:
        logger.error("session.review_failed", session_id=session_id, error=str(exc))
