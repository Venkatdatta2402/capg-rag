"""Quiz submission and regeneration endpoints.

Flow overview:
  1. POST /quiz/submit
       - Loads quiz questions + correct answers from InteractionStore (ES)
       - Grades each submitted option deterministically (selected == correct)
       - If wrong: judge LLM generates retrieval_feedback for that question
       - Updates the interaction document in ES with student_response + quiz.status
       - If FAILED: returns 3 prompt MCQ choices + aggregated retrieval_feedback

  2. POST /quiz/regenerate  (called after learner picks a prompt MCQ choice)
       - Re-runs the full pipeline with retrieval_hint appended to original query
       - Chosen prompt version bypasses the selector
       - Returns a fresh answer + new quiz_form
"""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.dependencies import get_interaction_store, get_pipeline, get_prompt_selector, get_session_store
from src.agents.judge import JudgeAgent
from src.models.interaction import StudentAnswer, StudentResponse
from src.models.query import QueryInput
from src.models.response import QuizForm
from src.orchestrator.pipeline import Pipeline
from src.prompt_service.selector import PromptSelector
from src.storage.interaction_store import InteractionStore
from src.storage.session_memory import SessionMemoryStore

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class AnswerSubmission(BaseModel):
    question_id: str        # "q1", "q2", "q3"
    selected_option: str    # "A" | "B" | "C" | "D"


class QuizSubmitRequest(BaseModel):
    quiz_id: str            # = session_id
    interaction_id: str     # from GenerationResponse — identifies the interaction in ES
    user_id: str
    answers: list[AnswerSubmission]


class QuestionVerdict(BaseModel):
    question_id: str
    question: str
    correct_answer: str
    selected_option: str
    verdict: str            # UNDERSTOOD | NOT_UNDERSTOOD
    rationale: str
    retrieval_feedback: str


class PromptChoice(BaseModel):
    version_id: str
    variant: str
    label: str
    template_preview: str


class QuizSubmitResponse(BaseModel):
    quiz_id: str
    overall: str            # PASSED | FAILED
    passed_count: int
    total_questions: int
    verdicts: list[QuestionVerdict]
    prompt_choices: list[PromptChoice] = []
    retrieval_hint: str = ""


class QuizRegenerateRequest(BaseModel):
    original_query: str
    user_id: str
    session_id: str
    chosen_prompt_version_id: str
    retrieval_hint: str = ""


class QuizRegenerateResponse(BaseModel):
    answer: str
    interaction_id: str
    prompt_version: str
    architecture: str
    retrieval_quality: str
    latency_ms: float
    quiz_form: QuizForm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VARIANT_LABELS = {
    "standard": "Clear step-by-step explanation",
    "remedial": "Extra support with analogies",
    "advanced": "Deeper connections and application",
}


def _build_prompt_choices(versions) -> list[PromptChoice]:
    return [
        PromptChoice(
            version_id=v.version_id,
            variant=v.variant,
            label=_VARIANT_LABELS.get(v.variant, v.variant.capitalize()),
            template_preview=v.template[:120],
        )
        for v in versions
    ]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/quiz/submit", response_model=QuizSubmitResponse)
async def submit_quiz(
    request: QuizSubmitRequest,
    pipeline: Pipeline = Depends(get_pipeline),
    interaction_store: InteractionStore = Depends(get_interaction_store),
    selector: PromptSelector = Depends(get_prompt_selector),
    session_store: SessionMemoryStore = Depends(get_session_store),
):
    """Grade MCQ answers deterministically; call judge LLM only on wrong answers."""
    session_doc = await interaction_store.get_session(request.quiz_id)
    if not session_doc:
        raise HTTPException(status_code=404, detail="Session not found.")

    interaction = next(
        (ix for ix in session_doc.interactions if ix.interaction_id == request.interaction_id),
        None,
    )
    if not interaction:
        raise HTTPException(status_code=404, detail="Interaction not found.")

    if interaction.quiz.status == "submitted":
        raise HTTPException(status_code=409, detail="Quiz already submitted.")

    grade = session_doc.grade
    explanation = interaction.model_answer
    topic = interaction.meta.topic
    judge: JudgeAgent = pipeline.judge_agent

    # Build lookup: question_id → QuizQuestion (with correct_answer)
    question_map = {q.question_id: q for q in interaction.quiz.questions}
    submitted = {a.question_id: a.selected_option for a in request.answers}

    verdicts: list[QuestionVerdict] = []
    student_answers: list[StudentAnswer] = []

    for qid, q in question_map.items():
        selected = submitted.get(qid, "")
        student_answers.append(StudentAnswer(question_id=qid, selected_option=selected))

        jv = await judge.grade_mcq(
            question=q.question,
            options=q.options,
            correct_answer=q.correct_answer,
            selected_option=selected,
            explanation=explanation,
            topic=topic,
        )

        verdicts.append(QuestionVerdict(
            question_id=qid,
            question=q.question,
            correct_answer=q.correct_answer,
            selected_option=selected,
            verdict=jv.verdict,
            rationale=jv.cot_reasoning,
            retrieval_feedback=jv.retrieval_feedback,
        ))

    passed = [v for v in verdicts if v.verdict == "UNDERSTOOD"]
    overall = "PASSED" if len(passed) >= 2 else "FAILED"
    score = len(passed)

    # Update ES interaction with student response
    student_response = StudentResponse(
        answers=student_answers,
        score=score,
        submitted_at=datetime.utcnow(),
    )
    await interaction_store.update_quiz_response(
        session_id=request.quiz_id,
        interaction_id=request.interaction_id,
        student_response=student_response,
        quiz_status="submitted",
    )

    # Mirror quiz outcome to runtime session memory for context agents
    await session_store.update_quiz_result(
        session_id=request.quiz_id,
        user_id=request.user_id,
        interaction_id=request.interaction_id,
        score=score,
        quiz_status="submitted",
    )

    prompt_choices: list[PromptChoice] = []
    retrieval_hint = ""

    if overall == "FAILED":
        feedbacks = [v.retrieval_feedback for v in verdicts if v.retrieval_feedback]
        retrieval_hint = "; ".join(feedbacks)

        # Set retry_mode on the persisted ContextObject for next pipeline run
        await session_store.set_retry_mode(
            session_id=request.quiz_id,
            user_id=request.user_id,
        )

        candidate_versions = await selector.select_candidates_from_feedback(
            grade=grade,
            retrieval_feedback=retrieval_hint or "general comprehension failure",
            n=3,
        )
        prompt_choices = _build_prompt_choices(candidate_versions)

    return QuizSubmitResponse(
        quiz_id=request.quiz_id,
        overall=overall,
        passed_count=score,
        total_questions=len(verdicts),
        verdicts=verdicts,
        prompt_choices=prompt_choices,
        retrieval_hint=retrieval_hint,
    )


@router.post("/quiz/regenerate", response_model=QuizRegenerateResponse)
async def regenerate_after_quiz(
    request: QuizRegenerateRequest,
    pipeline: Pipeline = Depends(get_pipeline),
):
    """Re-run the pipeline with the learner's chosen prompt and a retrieval hint."""
    enriched_query_text = request.original_query
    if request.retrieval_hint:
        enriched_query_text = (
            f"{request.original_query}\n"
            f"[Focus also on: {request.retrieval_hint}]"
        )

    query_input = QueryInput(
        query_text=enriched_query_text,
        user_id=request.user_id,
        session_id=request.session_id,
    )

    response = await pipeline.run_with_prompt_override(
        query_input=query_input,
        prompt_version_id=request.chosen_prompt_version_id,
    )

    return QuizRegenerateResponse(
        answer=response.answer_text,
        interaction_id=response.interaction_id,
        prompt_version=response.metadata.prompt_version,
        architecture=response.metadata.architecture,
        retrieval_quality=response.metadata.retrieval_quality_flag,
        latency_ms=response.metadata.latency_ms,
        quiz_form=response.quiz_form,
    )
