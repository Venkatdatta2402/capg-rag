"""Feedback and judge verdict models."""

from datetime import datetime

from pydantic import BaseModel, Field


class JudgeVerdict(BaseModel):
    """Output of the judge agent's evaluation."""

    verdict: str = ""                  # UNDERSTOOD | NOT_UNDERSTOOD
    cot_reasoning: str = ""
    question_asked: str = ""
    learner_response: str = ""
    retrieval_feedback: str = ""       # concept gap; used to generate follow-up question


class FeedbackRecord(BaseModel):
    """Governance feedback record written at session end."""

    user_id: str = ""
    session_id: str = ""
    grade: str = ""
    prompt_version: str = ""
    verdict: str = ""                  # "session_review"
    user_signal: str = ""              # "session_end"
    judge_cot_summary: str = ""        # governance_feedback from SessionReviewAgent
    retry_count: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)
