"""RAG Generation Agent.

Generates the final grounded answer. When generate_quiz=True (retrieval quality
GOOD and not retry mode), the LLM is given a `present_mcq` tool. It calls the
tool with 3 adversarial MCQ questions structured as JSON — no text parsing needed.

The tool call returns:
  - client-facing: QuizForm (questions + options, no correct answers)
  - server-side:   quiz_keys {question_id: correct_answer} stored in ES
"""

import time
from datetime import datetime

import structlog

from src.agents.base import BaseAgent
from src.llm.base import BaseLLMClient
from src.models.prompt import PromptVersion
from src.models.query import EnrichedQuery
from src.models.response import GenerationResponse, QuizForm, QuizQuestion, ResponseMetadata
from src.models.retrieval import RerankedChunk
from src.tools.present_mcq import PRESENT_MCQ_TOOL

logger = structlog.get_logger()

QUIZ_INSTRUCTION = """\

---

After your explanation, call the `present_mcq` tool with exactly 3 adversarial \
comprehension questions to expose gaps in the student's understanding.

ADVERSARIAL RULES:
1. Each question must use a NEW value or scenario not in your explanation.
2. A student who only memorised the explanation must fail each question.
3. Target the most likely misconceptions for this topic.
4. Prefer application/calculation questions.
5. Each question must have exactly 4 options (A, B, C, D). Only one is correct.
6. The 3 distractors must reflect plausible misconceptions, not random wrong answers."""


class RAGAgent(BaseAgent):
    """Generates grounded answers and optionally adversarial MCQ quiz questions."""

    def __init__(self, llm: BaseLLMClient):
        super().__init__(llm)

    async def run(
        self,
        enriched_query: EnrichedQuery,
        prompt_version: PromptVersion,
        chunks: list[RerankedChunk],
        generate_quiz: bool = False,
        quiz_id: str = "",
    ) -> tuple[GenerationResponse, dict[str, str]]:
        """Generate a grounded answer and optionally an adversarial MCQ quiz.

        Args:
            enriched_query: The enriched, rephrased query.
            prompt_version: The selected versioned prompt template.
            chunks: Top-K reranked chunks from retrieval.
            generate_quiz: When True, pass present_mcq tool to the LLM.
            quiz_id: Identifies the quiz in the ES document.

        Returns:
            (GenerationResponse, quiz_keys) tuple.
            quiz_keys: {question_id: correct_answer} — stored server-side in ES.
            Empty dict when generate_quiz=False.
        """
        start = time.time()

        context_text = "\n\n---\n\n".join(
            f"[Source: {c.source}]\n{c.text}" for c in chunks
        )

        system_prompt = prompt_version.template
        if generate_quiz:
            system_prompt += QUIZ_INSTRUCTION

        user_message = (
            f"Student's question: {enriched_query.rewritten_text}\n\n"
            f"Retrieved knowledge:\n{context_text}\n\n"
            f"Answer the question using ONLY the retrieved knowledge above."
        )

        if generate_quiz:
            answer_text, tool_calls = await self._llm.generate_with_tools(
                system_prompt=system_prompt,
                user_message=user_message,
                tools=[PRESENT_MCQ_TOOL],
            )
            quiz_form, quiz_keys = self._build_quiz_from_tool_call(tool_calls, quiz_id)
        else:
            answer_text = await self._llm.generate(
                system_prompt=system_prompt,
                user_message=user_message,
            )
            quiz_form = QuizForm(quiz_id=quiz_id, skipped=True, skip_reason="quiz not requested")
            quiz_keys = {}

        latency_ms = (time.time() - start) * 1000

        avg_score = sum(c.rerank_score for c in chunks) / len(chunks) if chunks else 0.0
        quality_flag = (
            "GOOD" if avg_score > 0.75 else "MARGINAL" if avg_score > 0.5 else "POOR"
        )
        cohort = "canary" if prompt_version.status == "candidate" else "control"

        metadata = ResponseMetadata(
            prompt_version=prompt_version.version_id,
            retrieved_sources=[c.source for c in chunks],
            retrieval_quality_score=round(avg_score, 4),
            retrieval_quality_flag=quality_flag,
            generation_model=self._llm.model,
            prompt_cohort=f"{prompt_version.version_id}_{cohort}",
            architecture="sequential",
            latency_ms=round(latency_ms, 2),
            timestamp=datetime.utcnow(),
        )

        logger.info(
            "rag_agent.generated",
            prompt_version=prompt_version.version_id,
            retrieval_quality=quality_flag,
            latency_ms=metadata.latency_ms,
            quiz_generated=generate_quiz,
            quiz_questions=len(quiz_form.questions),
        )

        return GenerationResponse(
            answer_text=answer_text.strip(),
            metadata=metadata,
            quiz_form=quiz_form,
        ), quiz_keys

    # ------------------------------------------------------------------
    # Tool call parser
    # ------------------------------------------------------------------

    def _build_quiz_from_tool_call(
        self,
        tool_calls: list[dict],
        quiz_id: str,
    ) -> tuple[QuizForm, dict[str, str]]:
        """Extract QuizForm and server-side keys from the present_mcq tool call."""
        mcq_call = next(
            (tc for tc in tool_calls if tc["name"] == "present_mcq"), None
        )
        if not mcq_call:
            logger.warning("rag_agent.present_mcq_tool_not_called")
            return QuizForm(quiz_id=quiz_id, skipped=True, skip_reason="tool_not_called"), {}

        raw_questions = mcq_call["arguments"].get("questions", [])
        questions: list[QuizQuestion] = []
        quiz_keys: dict[str, str] = {}

        for q in raw_questions:
            qid = q.get("question_id", "")
            question_text = q.get("question", "")
            options = q.get("options", [])
            correct_answer = q.get("correct_answer", "").upper()

            if not qid or not question_text or len(options) != 4 or not correct_answer:
                logger.warning("rag_agent.mcq_question_malformed", question_id=qid)
                continue

            questions.append(QuizQuestion(
                question_id=qid,
                question=question_text,
                options=options,
            ))
            quiz_keys[qid] = correct_answer

        if not questions:
            return QuizForm(quiz_id=quiz_id, skipped=True, skip_reason="tool_parse_failed"), {}

        logger.info("rag_agent.quiz_built", quiz_id=quiz_id, question_count=len(questions))
        return QuizForm(quiz_id=quiz_id, questions=questions), quiz_keys
