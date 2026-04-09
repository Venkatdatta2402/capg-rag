"""Evaluation Agent.

Runs asynchronously at session end (fire-and-forget via POST /session/end).
Reads interaction_id, question, model_answer, context_used from each interaction
and scores each on correctness, groundedness, answer_relevance, coherence, sufficiency.
One EvalResult written to ES per interaction.
"""

import structlog

from src.agents.base import BaseAgent
from src.llm.base import BaseLLMClient
from src.models.eval import EvalResult, InteractionEval
from src.storage.eval_store import EvalStore
from src.storage.interaction_store import InteractionStore

logger = structlog.get_logger()

_EVAL_PROMPT = """\
You are evaluating an AI tutor's response for a CBSE/NCERT student.

=== QUESTION ===
{question}

=== CONTEXT USED (retrieved chunks) ===
{context_block}

=== MODEL ANSWER ===
{model_answer}

Score the response on each dimension. Use the definitions strictly.

CORRECTNESS: 1 if the answer is factually correct based on the context, 0 if it contains errors.
GROUNDEDNESS: fraction of answer claims that are directly supported by the context (0.0–1.0).
ANSWER_RELEVANCE: how directly the answer addresses the question (0.0–1.0).
COHERENCE: logical flow and clarity of the answer (0.0–1.0).
SUFFICIENCY: how completely the answer covers what the question needs (0.0–1.0).
ERROR_TYPE: "none" if answer is good, "retrieval_failure" if context lacked needed info, "generation_failure" if context was adequate but answer was wrong or incomplete.

Respond in this exact format:
CORRECTNESS: <0 or 1>
GROUNDEDNESS: <float>
ANSWER_RELEVANCE: <float>
COHERENCE: <float>
SUFFICIENCY: <float>
ERROR_TYPE: <none | retrieval_failure | generation_failure>
"""


class EvalAgent(BaseAgent):
    """Evaluates each interaction in a session and writes scores to EvalStore."""

    def __init__(self, llm: BaseLLMClient):
        super().__init__(llm)

    async def run(self, **kwargs):
        pass  # primary interface is evaluate_session

    async def evaluate_session(
        self,
        session_id: str,
        interaction_store: InteractionStore,
        eval_store: EvalStore,
    ) -> None:
        session_doc = await interaction_store.get_session(session_id)
        if not session_doc or not session_doc.interactions:
            logger.info("eval_agent.skipped.no_interactions", session_id=session_id)
            return

        for ix in session_doc.interactions:
            context_block = "\n".join(
                f"[{c.rank}] {c.text}" for c in ix.context_used
            ) or "No context available."

            prompt = _EVAL_PROMPT.format(
                question=ix.question,
                context_block=context_block,
                model_answer=ix.model_answer,
            )

            try:
                response = await self._llm.generate(
                    system_prompt="You are an objective evaluator for an educational AI system.",
                    user_message=prompt,
                )
                interaction_eval = self._parse(ix.interaction_id, response)
            except Exception as exc:
                logger.error(
                    "eval_agent.llm_failed",
                    session_id=session_id,
                    interaction_id=ix.interaction_id,
                    error=str(exc),
                )
                interaction_eval = InteractionEval(
                    interaction_id=ix.interaction_id,
                    error_type="generation_failure",
                )

            result = EvalResult(session_id=session_id, interaction=interaction_eval)
            await eval_store.write(result)

        logger.info(
            "eval_agent.done",
            session_id=session_id,
            n=len(session_doc.interactions),
        )

    def _parse(self, interaction_id: str, response: str) -> InteractionEval:
        parsed: dict[str, str] = {}
        for line in response.strip().split("\n"):
            if ": " in line:
                key, _, value = line.partition(": ")
                parsed[key.strip().upper()] = value.strip()

        def _float(key: str, default: float = 0.0) -> float:
            try:
                return max(0.0, min(1.0, float(parsed.get(key, default))))
            except ValueError:
                return default

        def _int(key: str) -> float:
            try:
                return float(int(parsed.get(key, 0)))
            except ValueError:
                return 0.0

        error_type = parsed.get("ERROR_TYPE", "none").lower()
        if error_type not in {"none", "retrieval_failure", "generation_failure"}:
            error_type = "none"

        return InteractionEval(
            interaction_id=interaction_id,
            correctness=_int("CORRECTNESS"),
            groundedness=_float("GROUNDEDNESS"),
            answer_relevance=_float("ANSWER_RELEVANCE"),
            coherence=_float("COHERENCE"),
            sufficiency=_float("SUFFICIENCY"),
            error_type=error_type,
        )
