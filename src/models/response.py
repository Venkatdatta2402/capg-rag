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
    """A single MCQ quiz question — sent to the client as part of the form.

    options: flat strings e.g. ["A) 100 cm", "B) 10 cm", "C) 1000 cm", "D) 1 cm"]
    Correct answer is NOT included — it is stored server-side in ES via InteractionStore.
    """

    question_id: str                              # e.g. "q1", "q2", "q3"
    question: str
    options: list[str] = Field(default_factory=list)  # ["A) ...", "B) ...", "C) ...", "D) ..."]


class QuizForm(BaseModel):
    """The form embedded in the chat response.

    Contains questions + options only — correct answers are stored server-side.
    Submitting this form (question_id + selected_option per question) to
    POST /quiz/submit triggers MCQ grading.
    """

    quiz_id: str                                        # keyed by session_id
    questions: list[QuizQuestion] = Field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""                               # e.g. "retrieval_quality=POOR"


class GenerationResponse(BaseModel):
    """Full response from the RAG agent."""

    answer_text: str
    interaction_id: str = ""                            # ES interaction ID for quiz submit
    metadata: ResponseMetadata = Field(default_factory=ResponseMetadata)
    quiz_form: QuizForm = Field(default_factory=QuizForm)
