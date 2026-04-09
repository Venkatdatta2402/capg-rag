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
    get_eval_agent,
    get_eval_store,
    get_feedback_store,
    get_interaction_store,
    get_learner_profile_store,
    get_session_store,
    get_session_review_agent,
)
from src.agents.eval_agent import EvalAgent
from src.agents.session_review import SessionReviewAgent
from src.storage.eval_store import EvalStore
from src.storage.feedback_store import FeedbackStore
from src.storage.interaction_store import InteractionStore
from src.storage.learner_profile_store import LearnerProfileStore
from src.storage.session_memory import SessionMemoryStore

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
    profile_store: LearnerProfileStore = Depends(get_learner_profile_store),
    feedback_store: FeedbackStore = Depends(get_feedback_store),
    interaction_store: InteractionStore = Depends(get_interaction_store),
    review_agent: SessionReviewAgent = Depends(get_session_review_agent),
    eval_agent: EvalAgent = Depends(get_eval_agent),
    eval_store: EvalStore = Depends(get_eval_store),
):
    """Schedule end-of-session review and evaluation, return immediately.

    Background tasks:
      1. SessionReviewAgent — updates learner profile + writes governance feedback
      2. EvalAgent — scores each interaction and writes to session_evaluations ES
    """
    background_tasks.add_task(
        _run_session_review,
        session_id=request.session_id,
        user_id=request.user_id,
        session_store=session_store,
        profile_store=profile_store,
        feedback_store=feedback_store,
        interaction_store=interaction_store,
        review_agent=review_agent,
    )
    background_tasks.add_task(
        _run_eval,
        session_id=request.session_id,
        interaction_store=interaction_store,
        eval_agent=eval_agent,
        eval_store=eval_store,
    )
    logger.info("session.end_scheduled", session_id=request.session_id, user_id=request.user_id)
    return SessionEndResponse(status="review_scheduled")


async def _run_session_review(
    session_id: str,
    user_id: str,
    session_store: SessionMemoryStore,
    profile_store: LearnerProfileStore,
    feedback_store: FeedbackStore,
    interaction_store: InteractionStore,
    review_agent: SessionReviewAgent,
) -> None:
    """Background task: run the session review and update long-term profile."""
    try:
        # Guard: skip if no interactions were logged
        session_doc = await interaction_store.get_session(session_id)
        if not session_doc or not session_doc.interactions:
            logger.info("session.review_skipped.no_interactions", session_id=session_id)
            await session_store.delete(session_id)
            return

        profile_doc = await profile_store.get(user_id)

        # Review agent reads full interaction log directly from ES
        result = await review_agent.review(profile_doc, session_id, interaction_store)

        # Weighted-average update to ES learner profile
        await profile_store.update_from_review(user_id, session_id, result)

        # Send governance feedback
        feedback_record = review_agent.build_feedback_record(
            result,
            session_id=session_id,
            user_id=user_id,
            grade=profile_doc.grade,
        )
        await feedback_store.write(feedback_record)

        await session_store.delete(session_id)

        logger.info("session.review_complete", session_id=session_id, user_id=user_id)

    except Exception as exc:
        logger.error("session.review_failed", session_id=session_id, error=str(exc))


async def _run_eval(
    session_id: str,
    interaction_store: InteractionStore,
    eval_agent: EvalAgent,
    eval_store: EvalStore,
) -> None:
    """Background task: evaluate each interaction and write scores to ES."""
    try:
        await eval_agent.evaluate_session(session_id, interaction_store, eval_store)
    except Exception as exc:
        logger.error("session.eval_failed", session_id=session_id, error=str(exc))
