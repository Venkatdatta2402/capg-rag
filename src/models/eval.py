"""Evaluation result models.

One EvalResult document is written per interaction at session end.
ES index: session_evaluations
"""

from datetime import datetime

from pydantic import BaseModel, Field


class InteractionEval(BaseModel):
    interaction_id: str
    correctness: float = 0.0          # 0 or 1 — factual correctness vs context
    groundedness: float = 0.0         # 0.0–1.0 — answer grounded in context_used
    answer_relevance: float = 0.0     # 0.0–1.0 — relevance to the question
    coherence: float = 0.0            # 0.0–1.0 — logical coherence of answer
    sufficiency: float = 0.0          # 0.0–1.0 — completeness of answer
    error_type: str = "none"          # none | retrieval_failure | generation_failure


class EvalResult(BaseModel):
    """Root ES document — one per interaction evaluation."""
    session_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    interaction: InteractionEval
