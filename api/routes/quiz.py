"""Quiz submission endpoint.

POST /quiz/submit
  - Loads quiz questions + correct answers from InteractionStore (ES)
  - Grades each submitted option deterministically (selected == correct)
  - Judge LLM called only on wrong answers to generate retrieval_feedback
  - Updates the interaction document in ES with student_response + quiz.status
  - If PASSED: returns verdicts
  - If FAILED:
      1. Loads session memory from ES for judge context
      2. Judge generates a focused follow-up question targeting the knowledge gap
      3. set_retry_mode on ContextObject so selector picks a remedial prompt
      4. Runs the full pipeline with the judge's question
      5. Returns verdicts + follow-up answer + new quiz
"""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.dependencies import get_interaction_store, get_pipeline, get_session_store
from src.agents.judge import JudgeAgent
from src.models.interaction import StudentAnswer, StudentResponse
from src.models.query import QueryInput
from src.models.response import QuizForm
from src.orchestrator.pipeline import Pipeline
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


class QuizSubmitResponse(BaseModel):
    quiz_id: str
    overall: str            # PASSED | FAILED
    passed_count: int
    total_questions: int
    verdicts: list[QuestionVerdict]
    # FAILED only — judge-generated follow-up
    follow_up_question: str = ""
    follow_up_answer: str = ""
    follow_up_interaction_id: str = ""
    follow_up_quiz_form: QuizForm | None = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/quiz/submit", response_model=QuizSubmitResponse)
async def submit_quiz(
    request: QuizSubmitRequest,
    pipeline: Pipeline = Depends(get_pipeline),
    interaction_store: InteractionStore = Depends(get_interaction_store),
    session_store: SessionMemoryStore = Depends(get_session_store),
):
    """Grade MCQ answers deterministically; on FAILED run judge follow-up pipeline."""
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

    explanation = interaction.model_answer
    topic = interaction.meta.topic
    judge: JudgeAgent = pipeline.judge_agent

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

    await session_store.update_quiz_result(
        session_id=request.quiz_id,
        user_id=request.user_id,
        interaction_id=request.interaction_id,
        score=score,
        quiz_status="submitted",
    )

    if overall == "PASSED":
        return QuizSubmitResponse(
            quiz_id=request.quiz_id,
            overall=overall,
            passed_count=score,
            total_questions=len(verdicts),
            verdicts=verdicts,
        )

    # --- FAILED path ---

    # Load session memory for judge context
    session = await session_store.get(session_id=request.quiz_id, user_id=request.user_id)
    session_context = _format_session_context(session)

    # Aggregate retrieval feedback from wrong answers
    retrieval_feedback = "; ".join(v.retrieval_feedback for v in verdicts if v.retrieval_feedback)

    # Judge generates a focused question targeting the gap
    follow_up_question = await judge.generate_follow_up_question(
        topic=topic,
        retrieval_feedback=retrieval_feedback or "general comprehension gap",
        model_answer=explanation,
        session_context=session_context,
    )

    # Set retry_mode so selector picks a remedial prompt for the follow-up run
    await session_store.set_retry_mode(
        session_id=request.quiz_id,
        user_id=request.user_id,
    )

    # Re-run pipeline with judge's question (skips session context in transform_query)
    follow_up_input = QueryInput(
        query_text=follow_up_question,
        user_id=request.user_id,
        session_id=request.quiz_id,
    )
    follow_up_response = await pipeline.run_judge_followup(follow_up_input)

    return QuizSubmitResponse(
        quiz_id=request.quiz_id,
        overall=overall,
        passed_count=score,
        total_questions=len(verdicts),
        verdicts=verdicts,
        follow_up_question=follow_up_question,
        follow_up_answer=follow_up_response.answer_text,
        follow_up_interaction_id=follow_up_response.interaction_id,
        follow_up_quiz_form=follow_up_response.quiz_form,
    )


def _format_session_context(session) -> str:
    if not session:
        return ""
    parts = []
    past = session.summary_of_past
    if past.covered_topics:
        parts.append(f"Covered topics: {', '.join(past.covered_topics)}")
    if past.performance_trend != "unknown":
        parts.append(f"Performance trend: {past.performance_trend}")
    if session.recent_interactions:
        recent = [
            f"{ix.topic} (quiz={ix.quiz_status}, score={ix.score})"
            for ix in session.recent_interactions[-3:]
            if ix.topic
        ]
        if recent:
            parts.append(f"Recent: {'; '.join(recent)}")
    return "\n".join(parts)
