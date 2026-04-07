"""Quiz submission and regeneration endpoints.

Flow overview:
  1. POST /quiz/submit
       - Judge evaluates each submitted answer (adversarial CoT)
       - If FAILED: returns 3 prompt MCQ choices + aggregate retrieval_feedback
         so the learner (or UI) can pick a teaching style for a second attempt
       - Deletes quiz from store after evaluation (prevents replay)

  2. POST /quiz/regenerate  (called after learner picks a prompt MCQ choice)
       - Re-runs the full pipeline with:
           * retrieval_hint appended to original query (guides re-retrieval)
           * chosen prompt version bypassing the selector
       - Returns a fresh answer + new quiz_form
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.dependencies import get_pipeline, get_prompt_selector, get_quiz_store
from src.agents.arch_b.judge import SeparateJudgeAgent
from src.models.query import QueryInput
from src.models.response import QuizForm
from src.orchestrator.arch_b_pipeline import ArchBPipeline
from src.prompt_service.selector import PromptSelector
from src.storage.quiz_store import QuizStore

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class AnswerSubmission(BaseModel):
    question_id: str    # "q1", "q2", "q3"
    answer: str


class QuizSubmitRequest(BaseModel):
    quiz_id: str                        # session_id from the quiz_form
    user_id: str
    answers: list[AnswerSubmission]     # one entry per question answered


class QuestionVerdict(BaseModel):
    question_id: str
    question: str
    verdict: str                        # UNDERSTOOD | NOT_UNDERSTOOD
    rationale: str
    cot_reasoning: str
    retrieval_feedback: str             # what was missing; empty if UNDERSTOOD


class PromptChoice(BaseModel):
    """One MCQ option for teaching style selection after a failed quiz."""
    version_id: str
    variant: str                        # standard | remedial | advanced
    label: str                          # human-readable label shown in UI
    template_preview: str               # first 120 chars of the template


class QuizSubmitResponse(BaseModel):
    quiz_id: str
    overall: str                        # PASSED | FAILED
    passed_count: int
    total_questions: int
    verdicts: list[QuestionVerdict]
    # populated only when overall == FAILED
    prompt_choices: list[PromptChoice] = []
    retrieval_hint: str = ""            # aggregated retrieval_feedback for re-retrieval


class QuizRegenerateRequest(BaseModel):
    original_query: str
    user_id: str
    session_id: str
    chosen_prompt_version_id: str       # from PromptChoice.version_id
    retrieval_hint: str = ""            # from QuizSubmitResponse.retrieval_hint


class QuizRegenerateResponse(BaseModel):
    answer: str
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
    choices = []
    for v in versions:
        label = _VARIANT_LABELS.get(v.variant, v.variant.capitalize())
        choices.append(PromptChoice(
            version_id=v.version_id,
            variant=v.variant,
            label=label,
            template_preview=v.template[:120],
        ))
    return choices


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/quiz/submit", response_model=QuizSubmitResponse)
async def submit_quiz(
    request: QuizSubmitRequest,
    pipeline: ArchBPipeline = Depends(get_pipeline),
    quiz_store: QuizStore = Depends(get_quiz_store),
    selector: PromptSelector = Depends(get_prompt_selector),
):
    """Evaluate learner answers using the separate judge model (adversarial CoT).

    On FAILED:
      - Returns 3 prompt MCQ choices (fixed governance — one per teaching style)
      - Returns aggregated retrieval_feedback to guide re-retrieval in /quiz/regenerate
    On PASSED:
      - Returns verdicts only; prompt_choices and retrieval_hint are empty
    """
    if not isinstance(pipeline, ArchBPipeline):
        raise HTTPException(
            status_code=400,
            detail="Quiz submission is only supported in Architecture B.",
        )

    quiz_data = await quiz_store.load(request.quiz_id)
    if not quiz_data:
        raise HTTPException(status_code=404, detail="Quiz not found or already evaluated.")

    judge: SeparateJudgeAgent = pipeline.judge_agent
    explanation = quiz_data["explanation"]
    topic = quiz_data["topic"]
    grade = quiz_data.get("grade", "")
    expected_answers: dict[str, str] = quiz_data["answers"]

    submitted = {a.question_id: a.answer for a in request.answers}

    verdicts: list[QuestionVerdict] = []
    for qid, expected in expected_answers.items():
        learner_answer = submitted.get(qid, "")

        jv = await judge.evaluate(
            question=expected,
            learner_response=learner_answer,
            explanation=explanation,
            topic=topic,
        )

        rationale = (
            jv.cot_reasoning.split("RATIONALE:")[-1].split("\n")[0].strip()
            if "RATIONALE:" in jv.cot_reasoning else ""
        )

        verdicts.append(QuestionVerdict(
            question_id=qid,
            question=expected,
            verdict=jv.verdict,
            rationale=rationale,
            cot_reasoning=jv.cot_reasoning,
            retrieval_feedback=jv.retrieval_feedback,
        ))

    # Delete quiz after evaluation — prevent replay attacks
    await quiz_store.delete(request.quiz_id)

    passed = [v for v in verdicts if v.verdict == "UNDERSTOOD"]
    overall = "PASSED" if len(passed) >= 2 else "FAILED"

    prompt_choices: list[PromptChoice] = []
    retrieval_hint = ""

    if overall == "FAILED":
        # Aggregate retrieval feedback from all failed questions into one hint string
        feedbacks = [v.retrieval_feedback for v in verdicts if v.retrieval_feedback]
        retrieval_hint = "; ".join(feedbacks)

        # Feedback-aware selection: LLM scores all 6 seeded teaching styles against
        # the judge's retrieval feedback and returns the 3 most suitable ones.
        candidate_versions = await selector.select_candidates_from_feedback(
            grade=grade,
            retrieval_feedback=retrieval_hint or "general comprehension failure",
            n=3,
        )
        prompt_choices = _build_prompt_choices(candidate_versions)

    return QuizSubmitResponse(
        quiz_id=request.quiz_id,
        overall=overall,
        passed_count=len(passed),
        total_questions=len(verdicts),
        verdicts=verdicts,
        prompt_choices=prompt_choices,
        retrieval_hint=retrieval_hint,
    )


@router.post("/quiz/regenerate", response_model=QuizRegenerateResponse)
async def regenerate_after_quiz(
    request: QuizRegenerateRequest,
    pipeline: ArchBPipeline = Depends(get_pipeline),
):
    """Re-run the pipeline with the learner's chosen prompt and a retrieval hint.

    Called after the learner picks one of the 3 prompt MCQ choices returned
    by a failed /quiz/submit. The retrieval_hint (aggregated judge feedback)
    is appended to the original query so the pipeline retrieves more targeted
    content on this second attempt. The chosen prompt bypasses the selector.
    """
    if not isinstance(pipeline, ArchBPipeline):
        raise HTTPException(
            status_code=400,
            detail="Quiz regeneration is only supported in Architecture B.",
        )

    # Append retrieval hint to query so re-retrieval targets missing concepts
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
        prompt_version=response.metadata.prompt_version,
        architecture=response.metadata.architecture,
        retrieval_quality=response.metadata.retrieval_quality_flag,
        latency_ms=response.metadata.latency_ms,
        quiz_form=response.quiz_form,
    )
