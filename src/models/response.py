"""Generation response and metadata models."""

from datetime import datetime

from pydantic import BaseModel, Field


class ResponseMetadata(BaseModel):
    """Metadata attached to every generated response for downstream tracking."""

    prompt_version: str = ""
    retrieved_sources: list[str] = Field(default_factory=list)
    retrieval_quality_score: float = 0.0
    retrieval_quality_flag: str = ""   # GOOD, MARGINAL, POOR
    generation_model: str = ""
    prompt_cohort: str = ""
    architecture: str = ""
    latency_ms: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class QuizQuestion(BaseModel):
    """A single adversarial quiz question — sent to the client as part of the form."""

    question_id: str          # e.g. "q1", "q2", "q3"
    question: str
    mode: str = "quiz"        # quiz | explanation_prompt


class QuizForm(BaseModel):
    """The form embedded in the chat response.

    Contains only questions — expected answers are stored server-side.
    Submitting this form (all question_id + learner_answer pairs) to
    POST /quiz/submit triggers the judge model.
    """

    quiz_id: str                                        # keyed by session_id
    questions: list[QuizQuestion] = Field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""                               # e.g. "retrieval_quality=POOR"


class GenerationResponse(BaseModel):
    """Full response from the RAG agent."""

    answer_text: str
    metadata: ResponseMetadata = Field(default_factory=ResponseMetadata)
    quiz_form: QuizForm = Field(default_factory=QuizForm)
