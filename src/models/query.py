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
    """Output of the PromptSelector for one turn.

    All fields except retry_mode/retry_count are reasoned by the LLM in a single
    call that considers both the LearnerProfileDocument scores and the current query.

    retry_mode and retry_count are written by the judge path (set_retry_mode) and
    read back on the next turn.
    """

    # --- LLM-reasoned from profile + query ---
    grade: str = ""
    learning_styles: list[str] = Field(default_factory=list)   # learnstyle: tags
    softskills_strong: list[str] = Field(default_factory=list)  # softskill: tags
    softskills_weak: list[str] = Field(default_factory=list)    # softskill: tags
    topic_strength: str = "topic:weak"                          # "topic:strong" | "topic:weak"

    # --- written by judge path ---
    retry_mode: bool = False
    retry_count: int = 0
