"""Shared RAG Generation Agent — used by both architectures.

When generate_quiz=True (Arch B, retrieval GOOD, not retry mode), the answer
and 3 adversarial quiz questions are produced in ONE LLM call. The model has
simultaneous attention over the prompt, retrieved chunks, and the answer it
is writing — producing tighter, more targeted questions than any separate call.

Expected answers are parsed out and returned separately for server-side storage.
The client only ever sees the questions (QuizForm), never the expected answers.
"""

import time
import uuid
from datetime import datetime

import structlog

from src.agents.base import BaseAgent
from src.llm.base import BaseLLMClient
from src.models.prompt import PromptVersion
from src.models.query import EnrichedQuery
from src.models.response import GenerationResponse, QuizForm, QuizQuestion, ResponseMetadata
from src.models.retrieval import RerankedChunk

logger = structlog.get_logger()

QUIZ_SUFFIX = """\

---

After your explanation, generate exactly 3 adversarial comprehension questions \
to expose gaps in the student's understanding.

ADVERSARIAL RULES:
1. Each question must use a NEW value or scenario not in your explanation.
2. A student who only memorised the explanation must fail each question.
3. Target the most likely misconceptions for this topic.
4. Prefer application/calculation questions. Use explanation-prompt only if \
the topic has no computable answer.
5. Provide the correct expected answer for each question (stored server-side, \
not shown to the student).

Output format (after your explanation, on new lines):
QUIZ_Q1: <question>
QUIZ_A1: <expected answer>
QUIZ_Q2: <question>
QUIZ_A2: <expected answer>
QUIZ_Q3: <question>
QUIZ_A3: <expected answer>"""


class RAGAgent(BaseAgent):
    """Generates the final grounded answer — and optionally 3 adversarial quiz questions
    in the same LLM call when conditions are met."""

    def __init__(self, llm: BaseLLMClient):
        super().__init__(llm)

    async def run(
        self,
        enriched_query: EnrichedQuery,
        prompt_version: PromptVersion,
        chunks: list[RerankedChunk],
        architecture: str = "",
        generate_quiz: bool = False,
        quiz_id: str = "",
    ) -> tuple[GenerationResponse, dict[str, str]]:
        """Generate a grounded answer and optionally an adversarial quiz.

        Args:
            enriched_query: The enriched, rephrased query.
            prompt_version: The selected versioned prompt template.
            chunks: Top-K reranked chunks from retrieval.
            architecture: "A" or "B" for metadata tagging.
            generate_quiz: When True, append quiz generation to the same prompt.
            quiz_id: Identifier to key expected answers in the quiz store.

        Returns:
            (GenerationResponse, expected_answers) tuple.
            expected_answers: {question_id: expected_answer_text} — store server-side.
            Empty dict when generate_quiz=False.
        """
        start = time.time()

        context_text = "\n\n---\n\n".join(
            f"[Source: {c.source}]\n{c.text}" for c in chunks
        )

        system_prompt = prompt_version.template
        if generate_quiz:
            system_prompt += QUIZ_SUFFIX

        user_message = (
            f"Student's question: {enriched_query.rewritten_text}\n\n"
            f"Retrieved knowledge:\n{context_text}\n\n"
            f"Answer the question using ONLY the retrieved knowledge above."
        )

        raw = await self._llm.generate(
            system_prompt=system_prompt,
            user_message=user_message,
        )

        latency_ms = (time.time() - start) * 1000

        # Split answer from quiz block
        answer_text, quiz_form, expected_answers = self._parse_output(
            raw, generate_quiz, quiz_id
        )

        # Retrieval quality
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
            architecture=architecture,
            latency_ms=round(latency_ms, 2),
            timestamp=datetime.utcnow(),
        )

        logger.info(
            "rag_agent.generated",
            prompt_version=prompt_version.version_id,
            retrieval_quality=quality_flag,
            latency_ms=metadata.latency_ms,
            quiz_generated=generate_quiz,
        )

        return GenerationResponse(
            answer_text=answer_text,
            metadata=metadata,
            quiz_form=quiz_form,
        ), expected_answers

    def _parse_output(
        self, raw: str, generate_quiz: bool, quiz_id: str
    ) -> tuple[str, QuizForm, dict[str, str]]:
        """Split the raw LLM output into answer text, quiz form, and expected answers."""
        if not generate_quiz:
            return raw.strip(), QuizForm(quiz_id=quiz_id, skipped=True, skip_reason="quiz not requested"), {}

        # Split on first QUIZ_Q1 marker
        parts = raw.split("QUIZ_Q1:", 1)
        answer_text = parts[0].strip()

        if len(parts) < 2:
            # Model didn't produce quiz block despite instructions
            logger.warning("rag_agent.quiz_parse_failed")
            return answer_text, QuizForm(quiz_id=quiz_id, skipped=True, skip_reason="parse_failed"), {}

        quiz_block = "QUIZ_Q1:" + parts[1]
        parsed = {}
        for line in quiz_block.strip().split("\n"):
            if ":" in line:
                key, _, value = line.partition(":")
                parsed[key.strip().upper()] = value.strip()

        questions = []
        expected_answers = {}

        for i in (1, 2, 3):
            q_key = f"QUIZ_Q{i}"
            a_key = f"QUIZ_A{i}"
            question_text = parsed.get(q_key, "")
            expected = parsed.get(a_key, "")

            if not question_text:
                continue

            qid = f"q{i}"
            questions.append(QuizQuestion(question_id=qid, question=question_text, mode="quiz"))
            expected_answers[qid] = expected

        quiz_form = QuizForm(quiz_id=quiz_id, questions=questions)

        logger.info("rag_agent.quiz_parsed", quiz_id=quiz_id, question_count=len(questions))
        return answer_text, quiz_form, expected_answers
