"""Chat endpoint — main interaction route for the RAG system."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.dependencies import get_pipeline
from src.models.query import QueryInput
from src.models.response import QuizForm
from src.orchestrator.base import BaseOrchestrator

router = APIRouter()


class ChatRequest(BaseModel):
    query: str
    user_id: str
    session_id: str


class ChatResponse(BaseModel):
    answer: str
    prompt_version: str
    architecture: str
    retrieval_quality: str
    latency_ms: float
    quiz_form: QuizForm     # questions only — expected answers stored server-side


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    pipeline: BaseOrchestrator = Depends(get_pipeline),
):
    """Process a learner query through the active architecture pipeline.

    When quiz_form.skipped is False, the response includes 3 adversarial
    comprehension questions. Submit learner answers to POST /quiz/submit
    to trigger judge evaluation.
    """
    query_input = QueryInput(
        query_text=request.query,
        user_id=request.user_id,
        session_id=request.session_id,
    )

    response = await pipeline.run(query_input)

    return ChatResponse(
        answer=response.answer_text,
        prompt_version=response.metadata.prompt_version,
        architecture=response.metadata.architecture,
        retrieval_quality=response.metadata.retrieval_quality_flag,
        latency_ms=response.metadata.latency_ms,
        quiz_form=response.quiz_form,
    )
