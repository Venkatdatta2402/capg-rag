"""Session state models."""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.models.query import ContextObject  # noqa: E402


class RecentInteraction(BaseModel):
    """One interaction recorded in the runtime session window (last 5)."""

    interaction_id: str = ""
    question: str = ""
    model_answer: str = ""
    topic: str = ""
    prompt_version: str = ""
    retry_count: int = 0
    quiz_status: str = "ignored"    # "ignored" | "attempted" | "submitted"
    score: int = 0                  # correct answers out of quiz questions (0 if not submitted)


class PastSummary(BaseModel):
    """Condensed summary of interactions that have scrolled off the recent window."""

    covered_topics: list[str] = Field(default_factory=list)
    common_errors: list[str] = Field(default_factory=list)      # questions answered wrong
    key_misconceptions: list[str] = Field(default_factory=list) # same as common_errors (no LLM here)
    performance_trend: str = "unknown"  # "improving" | "stable" | "declining" | "unknown"
    archived_scores: list[int] = Field(default_factory=list)    # quiz scores for trend computation


class SessionState(BaseModel):
    """Short-term session memory backed by Elasticsearch."""

    session_id: str
    user_id: str
    retry_count: int = 0
    context_object: ContextObject | None = None   # built once, mutated on retry
    recent_interactions: list[RecentInteraction] = Field(default_factory=list)
    summary_of_past: PastSummary = Field(default_factory=PastSummary)
