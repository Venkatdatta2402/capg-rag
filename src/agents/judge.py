"""Judge Agent — grades MCQ answers and generates retrieval feedback for wrong answers.

Responsibility split:
- Question + option generation → RAGAgent (tool call, structured output)
- MCQ grading                  → deterministic (selected == correct_answer)
- Retrieval feedback           → JudgeAgent LLM (only called on wrong answers)

Running on a separate, dedicated model prevents leniency bias.
"""

import structlog

from src.agents.base import BaseAgent
from src.llm.base import BaseLLMClient
from src.models.feedback import JudgeVerdict

logger = structlog.get_logger()

_FEEDBACK_SYSTEM_PROMPT = """\
You are a diagnostic evaluator for an educational AI system.

A student answered an MCQ question incorrectly. Your job is to identify what
specific concept or detail was missing from the original explanation that caused
this misunderstanding.

Respond in this exact format:
RATIONALE: <one-line explanation of why the wrong option was chosen>
RETRIEVAL_FEEDBACK: <retrieval query fragment that would fetch the missing concept, \
e.g. "definition and mechanism of osmosis in plants">
"""


class JudgeAgent(BaseAgent):
    """Grades MCQ answers deterministically; runs LLM only to generate retrieval feedback."""

    def __init__(self, llm: BaseLLMClient):
        super().__init__(llm)

    async def grade_mcq(
        self,
        question: str,
        options: list[str],
        correct_answer: str,
        selected_option: str,
        explanation: str,
        topic: str,
    ) -> JudgeVerdict:
        """Grade one MCQ answer.

        Verdict is deterministic. LLM is called only when the answer is wrong
        to produce retrieval_feedback for re-retrieval.

        Args:
            question: The MCQ question text.
            options: The 4 option strings e.g. ["A) 100 cm", "B) 10 cm", ...].
            correct_answer: The correct option letter e.g. "B".
            selected_option: The student's selected option letter.
            explanation: Original explanation from RAG agent (context for feedback).
            topic: Topic being assessed.
        """
        understood = selected_option.upper() == correct_answer.upper()

        if understood:
            verdict = JudgeVerdict(
                verdict="UNDERSTOOD",
                cot_reasoning="",
                check_mode="mcq",
                question_asked=question,
                learner_response=selected_option,
                retrieval_feedback="",
            )
            logger.info("judge.graded", verdict="UNDERSTOOD", topic=topic)
            return verdict

        # Wrong answer — call LLM to diagnose the gap
        rationale, retrieval_feedback = await self._generate_feedback(
            question=question,
            options=options,
            correct_answer=correct_answer,
            selected_option=selected_option,
            explanation=explanation,
            topic=topic,
        )

        verdict = JudgeVerdict(
            verdict="NOT_UNDERSTOOD",
            cot_reasoning=rationale,
            check_mode="mcq",
            question_asked=question,
            learner_response=selected_option,
            retrieval_feedback=retrieval_feedback,
        )
        logger.info("judge.graded", verdict="NOT_UNDERSTOOD", topic=topic)
        return verdict

    async def _generate_feedback(
        self,
        question: str,
        options: list[str],
        correct_answer: str,
        selected_option: str,
        explanation: str,
        topic: str,
    ) -> tuple[str, str]:
        """Call LLM to diagnose why the wrong option was chosen."""
        options_text = "\n".join(options)
        user_msg = (
            f"Topic: {topic}\n"
            f"Explanation given to the student:\n{explanation}\n\n"
            f"Question: {question}\n"
            f"Options:\n{options_text}\n\n"
            f"Correct answer: {correct_answer}\n"
            f"Student selected: {selected_option}\n\n"
            f"Diagnose the gap."
        )

        try:
            response = await self._llm.generate(
                system_prompt=_FEEDBACK_SYSTEM_PROMPT,
                user_message=user_msg,
            )
            parsed = {}
            for line in response.strip().split("\n"):
                if ": " in line:
                    key, _, value = line.partition(": ")
                    parsed[key.strip().upper()] = value.strip()

            rationale = parsed.get("RATIONALE", "")
            raw_feedback = parsed.get("RETRIEVAL_FEEDBACK", "")
            return rationale, raw_feedback
        except Exception as exc:
            logger.warning("judge.feedback_generation_failed", error=str(exc))
            return "", ""
