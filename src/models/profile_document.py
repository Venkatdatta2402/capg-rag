"""Elasticsearch learner profile document model.

Each skill / learning style entry carries a running weighted-average score
and the total observation count used to compute it.

score > 0  → strength
score <= 0 → weakness / gap

Stored in the ES index: learner_profiles (one doc per learner_id).
"""

from pydantic import BaseModel, Field


class SkillEntry(BaseModel):
    score: float = 0.0
    count: int = 0


class LearnerProfileDocument(BaseModel):
    """ES storage model for the long-term learner profile."""

    learner_id: str
    grade: str = ""
    board: str = "CBSE"

    # Keys are bare skill names (no namespace prefix):
    #   softskills       → "decomposition", "working_memory", ...
    #   learning_style   → "example_driven", "step_by_step", ...
    #   technical_skills → free-text topic names ("algebra", "thermodynamics", ...)
    softskills: dict[str, SkillEntry] = Field(default_factory=dict)
    learning_style: dict[str, SkillEntry] = Field(default_factory=dict)
    technical_skills: dict[str, SkillEntry] = Field(default_factory=dict)

    last_updated_session_id: str = ""
