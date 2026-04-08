"""Elasticsearch interaction document models.

One document per session. Each question-answer turn appends a new Interaction
to the interactions array. Quiz submission updates the matching interaction
in-place via interaction_id.

ES index: session_interactions
"""

from datetime import datetime

from pydantic import BaseModel, Field


class QuizQuestion(BaseModel):
    """Full quiz question as stored in ES (correct_answer included).

    options: flat strings e.g. ["A) 100 cm", "B) 10 cm", "C) 1000 cm", "D) 1 cm"]
    correct_answer: option letter e.g. "B"
    """
    question_id: str
    question: str
    options: list[str]          # ["A) ...", "B) ...", "C) ...", "D) ..."]
    correct_answer: str         # "A" | "B" | "C" | "D"


class QuizData(BaseModel):
    quiz_id: str
    questions: list[QuizQuestion] = Field(default_factory=list)
    status: str = "ignored"     # ignored | attempted | submitted


class StudentAnswer(BaseModel):
    question_id: str
    selected_option: str        # A | B | C | D


class StudentResponse(BaseModel):
    answers: list[StudentAnswer] = Field(default_factory=list)
    score: int = 0              # number of correct answers
    submitted_at: datetime | None = None


class ContextChunk(BaseModel):
    chunk_id: str
    text: str
    rank: int


class InteractionMeta(BaseModel):
    subject: str = ""
    topic: str = ""
    difficulty: str = ""        # easy | medium | hard (derived from prompt variant)


class Interaction(BaseModel):
    interaction_id: str
    question: str
    model_answer: str
    quiz: QuizData
    student_response: StudentResponse | None = None
    context_used: list[ContextChunk] = Field(default_factory=list)
    meta: InteractionMeta = Field(default_factory=InteractionMeta)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SessionInteractionDocument(BaseModel):
    """Root ES document — one per session, interactions grow as the session progresses."""
    session_id: str
    user_id: str
    grade: str = ""
    started_at: datetime = Field(default_factory=datetime.utcnow)
    interactions: list[Interaction] = Field(default_factory=list)
