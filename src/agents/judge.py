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

_FOLLOW_UP_SYSTEM_PROMPT = """\
You are diagnosing a knowledge gap after a student failed a quiz.
Generate ONE focused question phrased naturally as a student asking their tutor,
targeting the identified gap so the tutor re-explains exactly what was missed.
Respond with just the question — no preamble, no label.
"""

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

    async def generate_follow_up_question(
        self,
        topic: str,
        retrieval_feedback: str,
        model_answer: str,
        session_context: str = "",
    ) -> str:
        """Generate a student-phrased follow-up question targeting the knowledge gap.

        Called on quiz FAILED. The question is fed directly into the pipeline
        as the next query, with retry_mode already set on the ContextObject.
        """
        session_block = f"\nSession history:\n{session_context}" if session_context else ""
        user_msg = (
            f"Topic: {topic}\n"
            f"Explanation the student received:\n{model_answer}\n\n"
            f"Identified gaps from wrong answers: {retrieval_feedback}"
            f"{session_block}"
        )
        try:
            question = await self._llm.generate(
                system_prompt=_FOLLOW_UP_SYSTEM_PROMPT,
                user_message=user_msg,
            )
            question = question.strip().strip('"')
            logger.info("judge.follow_up_generated", topic=topic, question=question[:80])
            return question
        except Exception as exc:
            logger.warning("judge.follow_up_failed", error=str(exc))
            # Fallback: generic re-ask using the retrieval feedback
            return f"Can you explain {retrieval_feedback or topic} in more detail?"

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
