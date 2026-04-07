"""Query-related data models used across both architectures."""

from pydantic import BaseModel, Field


class QueryInput(BaseModel):
    """Raw input from the learner."""

    query_text: str
    user_id: str
    session_id: str


class EnrichedQuery(BaseModel):
    """Query after context enrichment and rephrasing for retrieval."""

    original_text: str
    rewritten_text: str
    keywords: list[str] = Field(default_factory=list)
    subject: str = ""
    topic: str = ""
    sub_topic: str = ""
    query_type: str = ""  # conceptual, procedural, conceptual_and_procedural


class ContextObject(BaseModel):
    """Structured context assembled for prompt selection and generation."""

    learner_grade: str = ""
    comprehension_level: str = ""  # low, low-medium, medium, high
    learning_style: str = ""       # example-driven, text-heavy, visual, etc.
    weak_areas: list[str] = Field(default_factory=list)
    session_understood: list[str] = Field(default_factory=list)
    current_topic: str = ""
    query_type: str = ""
    retry_mode: bool = False
    retry_count: int = 0
    failed_sub_concept: str = ""
