"""Learner profile and session state models."""

from pydantic import BaseModel, Field


class LearnerProfile(BaseModel):
    """Long-term learner profile stored in PostgreSQL JSONB."""

    user_id: str
    grade: str = ""                            # e.g. "Class 3"
    board: str = "CBSE"
    comprehension_score: float = 5.0           # 0-10 scale
    learning_style: str = "example-driven"
    weak_areas: list[str] = Field(default_factory=list)
    mastered_concepts: list[str] = Field(default_factory=list)


class SessionState(BaseModel):
    """Short-term session memory stored in Redis."""

    session_id: str
    user_id: str
    topics_discussed: list[str] = Field(default_factory=list)
    concepts_understood: list[str] = Field(default_factory=list)
    active_knowledge_gaps: list[str] = Field(default_factory=list)
    retry_count: int = 0
    escalation_mode: str = "normal"  # normal, quiz_only
