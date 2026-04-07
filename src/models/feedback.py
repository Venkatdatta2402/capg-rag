"""Feedback and judge verdict models."""

from datetime import datetime

from pydantic import BaseModel, Field


class JudgeVerdict(BaseModel):
    """Output of the judge agent's evaluation."""

    verdict: str = ""                  # UNDERSTOOD, NOT_UNDERSTOOD
    cot_reasoning: str = ""            # Full chain-of-thought trace
    check_mode: str = ""               # quiz, explanation_prompt, confidence_check
    question_asked: str = ""
    learner_response: str = ""
    retrieval_feedback: str = ""       # What was missing; used to guide re-retrieval


class FeedbackRecord(BaseModel):
    """Structured feedback record written to the Feedback Store after each interaction."""

    record_id: str = ""
    user_id: str = ""
    session_id: str = ""
    prompt_version: str = ""
    prompt_cohort: str = ""
    architecture: str = ""
    topic: str = ""
    grade: str = ""
    verdict: str = ""
    retrieval_quality_score: float = 0.0
    retrieval_quality_flag: str = ""
    retry_count: int = 0
    user_signal: str = ""              # quiz_correct, quiz_incorrect, etc.
    judge_cot_summary: str = ""
    generation_model: str = ""
    latency_ms: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)
