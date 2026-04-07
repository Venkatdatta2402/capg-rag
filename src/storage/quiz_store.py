"""Quiz store — holds expected answers server-side, keyed by quiz_id (= session_id).

Expected answers are NEVER sent to the client. Only questions travel in the
QuizForm. On /quiz/submit the judge loads expected answers from here to evaluate.
"""

import structlog

logger = structlog.get_logger()

# In-memory store; production uses Redis with TTL matching session expiry.
_store: dict[str, dict] = {}
# Structure:
# {
#   quiz_id: {
#     "explanation": str,
#     "topic": str,
#     "grade": str,
#     "answers": {question_id: expected_answer_text}
#   }
# }


class QuizStore:

    async def save(
        self,
        quiz_id: str,
        explanation: str,
        topic: str,
        grade: str,
        expected_answers: dict[str, str],
    ) -> None:
        """Store expected answers after quiz generation."""
        _store[quiz_id] = {
            "explanation": explanation,
            "topic": topic,
            "grade": grade,
            "answers": expected_answers,
        }
        logger.info("quiz_store.saved", quiz_id=quiz_id, question_count=len(expected_answers))

    async def load(self, quiz_id: str) -> dict | None:
        """Load quiz data for judge evaluation. Returns None if not found."""
        data = _store.get(quiz_id)
        if not data:
            logger.warning("quiz_store.not_found", quiz_id=quiz_id)
        return data

    async def delete(self, quiz_id: str) -> None:
        """Remove quiz after evaluation to prevent replay."""
        _store.pop(quiz_id, None)
