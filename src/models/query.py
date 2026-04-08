"""Query-related data models."""

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
    """Learner profile and retry state for prompt selection and grounded generation.

    Only carries long-term profile signals and retry state — query-technical
    fields (topic, query_type, keywords) live in EnrichedQuery.
    """

    # --- learner profile signals (from long-term memory) ---
    learner_grade: str = ""
    learning_styles: list[str] = Field(default_factory=list)   # learnstyle: tags
    technically_strong_areas: list[str] = Field(default_factory=list)
    technically_weak_areas: list[str] = Field(default_factory=list)
    softskills_strong_areas: list[str] = Field(default_factory=list)  # softskill: tags
    softskills_weak_areas: list[str] = Field(default_factory=list)    # softskill: tags

    # --- retry state ---
    retry_mode: bool = False
    retry_count: int = 0
