"""Architecture A: Separate Judge Agent.

In Arch A, the judge runs on its own model (typically a lighter model)
separate from the generation model. It uses adversarial CoT evaluation.
"""

import structlog

from src.agents.base import BaseAgent
from src.llm.base import BaseLLMClient
from src.models.feedback import JudgeVerdict

logger = structlog.get_logger()

JUDGE_SYSTEM_PROMPT = """\
You are a rigorous academic evaluator. Your job is to find gaps in understanding,
not to confirm that the learner did well.

Assume partial understanding is NOT sufficient. Only mark UNDERSTOOD if the learner
demonstrates the core concept without prompting.

You MUST follow this chain-of-thought evaluation process:

Step 1: What was the core concept the learner needed to demonstrate?
Step 2: What did the learner actually say or answer?
Step 3: Does the response contain the key concept? What is missing?
Step 4: Is this a genuine understanding signal or a surface guess?
Step 5: Verdict -- UNDERSTOOD or NOT_UNDERSTOOD, and why.

Respond in this exact format:
COT_STEP_1: <...>
COT_STEP_2: <...>
COT_STEP_3: <...>
COT_STEP_4: <...>
VERDICT: <UNDERSTOOD|NOT_UNDERSTOOD>
RATIONALE: <one-line summary>
"""

COMPREHENSION_CHECK_PROMPT = """\
Based on the explanation just given to the learner, generate a comprehension check question.

The explanation was about: {topic}
Grade level: {grade}
Explanation given: {explanation}

Choose one of these modes based on complexity:
- MODE_A (Quiz): A specific question that requires applying the concept
- MODE_B (Explanation): Ask the learner to explain in their own words
- MODE_C (Confidence): Simple yes/no understanding check (only for very simple confirmations)

Prefer MODE_A for maximum reliability.

Respond with:
CHECK_MODE: <quiz|explanation_prompt|confidence_check>
QUESTION: <the comprehension check question>
"""


class SeparateJudgeAgent(BaseAgent):
    """Judge agent running on its own model (Architecture A).

    Uses adversarial CoT to counter leniency bias.
    """

    def __init__(self, llm: BaseLLMClient):
        super().__init__(llm)

    async def generate_check(self, topic: str, grade: str, explanation: str) -> dict:
        """Generate a comprehension check question.

        Returns:
            dict with keys: check_mode, question
        """
        prompt = COMPREHENSION_CHECK_PROMPT.format(
            topic=topic, grade=grade, explanation=explanation
        )
        response = await self._llm.generate(
            system_prompt="You are an educational assessment designer.",
            user_message=prompt,
        )
        return self._parse_check(response)

    async def evaluate(
        self, question: str, learner_response: str, explanation: str, topic: str
    ) -> JudgeVerdict:
        """Evaluate the learner's response using adversarial CoT.

        Args:
            question: The comprehension check question that was asked.
            learner_response: What the learner answered.
            explanation: The original explanation given.
            topic: The topic being assessed.
        """
        user_msg = (
            f"Topic: {topic}\n"
            f"Explanation given: {explanation}\n"
            f"Comprehension question: {question}\n"
            f"Learner's response: {learner_response}\n\n"
            f"Evaluate using the required chain-of-thought process."
        )

        response = await self._llm.generate(
            system_prompt=JUDGE_SYSTEM_PROMPT,
            user_message=user_msg,
        )

        verdict = self._parse_verdict(response, question, learner_response)
        logger.info("judge.evaluated", verdict=verdict.verdict, topic=topic)
        return verdict

    def _parse_check(self, response: str) -> dict:
        parsed = {}
        for line in response.strip().split("\n"):
            if ":" in line:
                key, _, value = line.partition(":")
                parsed[key.strip().upper()] = value.strip()
        return {
            "check_mode": parsed.get("CHECK_MODE", "quiz"),
            "question": parsed.get("QUESTION", "Can you explain what you learned?"),
        }

    def _parse_verdict(
        self, response: str, question: str, learner_response: str
    ) -> JudgeVerdict:
        parsed = {}
        for line in response.strip().split("\n"):
            if ":" in line:
                key, _, value = line.partition(":")
                parsed[key.strip().upper()] = value.strip()

        return JudgeVerdict(
            verdict=parsed.get("VERDICT", "NOT_UNDERSTOOD"),
            cot_reasoning=response,
            check_mode=parsed.get("CHECK_MODE", ""),
            question_asked=question,
            learner_response=learner_response,
        )
