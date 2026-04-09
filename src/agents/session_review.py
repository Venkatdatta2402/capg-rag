"""Session Review Agent.

Runs asynchronously at the end of a session (fire-and-forget via POST /session/end).
Reads the full interaction log from the InteractionStore (ES) — richer than session_memory
since it includes quiz questions, correct answers, student responses, and scores.

Outputs:
  - Updated technically_strong / technically_weak / softskills_strong / softskills_weak areas
  - Inferred or confirmed learning_styles
  - Governance feedback for the prompt governance pipeline
"""

import structlog

from src.agents.base import BaseAgent
from src.llm.base import BaseLLMClient
from src.models.feedback import FeedbackRecord
from src.models.profile_document import LearnerProfileDocument
from src.storage.interaction_store import InteractionStore

logger = structlog.get_logger()


SESSION_REVIEW_PROMPT = """\
You are reviewing a student's learning session to update their long-term profile.

=== INTERACTION LOG ({n} interactions) ===
{interaction_block}

=== CURRENT LEARNER PROFILE ===
Grade: {grade}
Learning styles: {current_learning_styles}
Technical strengths: {current_strong}
Technical weaknesses: {current_weak}
Soft skill strengths: {current_softskills_strong}
Soft skill weaknesses: {current_softskills_weak}

Based on the full interaction log, provide updated profile fields.

For TECHNICALLY_STRONG and TECHNICALLY_WEAK use topic/concept names (free text).

For SOFTSKILLS_STRONG and SOFTSKILLS_WEAK choose only from this fixed tag set:
  Reasoning:         softskill:decomposition, softskill:abstraction, softskill:pattern_mapping
  Cognitive Control: softskill:working_memory, softskill:attention_control, softskill:process_discipline
  Metacognition:     softskill:error_detection, softskill:confidence_calibration, softskill:reflection

For LEARNING_STYLES choose ALL that apply (comma-separated):
  Input:       learnstyle:visual, learnstyle:textual, learnstyle:example_driven, learnstyle:abstract_first
  Interaction: learnstyle:guided, learnstyle:exploratory, learnstyle:step_by_step, learnstyle:challenge_based
  Feedback:    learnstyle:immediate_feedback, learnstyle:delayed_reflection, learnstyle:hint_sensitive

Respond in this exact format:
TECHNICALLY_STRONG: <comma-separated topics, or "none">
TECHNICALLY_WEAK: <comma-separated topics, or "none">
SOFTSKILLS_STRONG: <comma-separated tags, or "none">
SOFTSKILLS_WEAK: <comma-separated tags, or "none">
LEARNING_STYLES: <comma-separated tags, or "none">
GOVERNANCE_FEEDBACK: <1-2 sentences on what teaching approaches worked or didn't this session>
"""

_VALID_SOFTSKILL_TAGS = {
    "softskill:decomposition", "softskill:abstraction", "softskill:pattern_mapping",
    "softskill:working_memory", "softskill:attention_control", "softskill:process_discipline",
    "softskill:error_detection", "softskill:confidence_calibration", "softskill:reflection",
}
_VALID_LEARNSTYLE_TAGS = {
    "learnstyle:visual", "learnstyle:textual", "learnstyle:example_driven", "learnstyle:abstract_first",
    "learnstyle:guided", "learnstyle:exploratory", "learnstyle:step_by_step", "learnstyle:challenge_based",
    "learnstyle:immediate_feedback", "learnstyle:delayed_reflection", "learnstyle:hint_sensitive",
}


class SessionReviewResult:
    def __init__(
        self,
        technically_strong: list[str],
        technically_weak: list[str],
        softskills_strong: list[str],
        softskills_weak: list[str],
        learning_styles: list[str],
        governance_feedback: str,
    ):
        self.technically_strong = technically_strong
        self.technically_weak = technically_weak
        self.softskills_strong = softskills_strong
        self.softskills_weak = softskills_weak
        self.learning_styles = learning_styles
        self.governance_feedback = governance_feedback


class SessionReviewAgent(BaseAgent):
    """Reviews the full session interaction log and produces an updated learner profile."""

    def __init__(self, llm: BaseLLMClient):
        super().__init__(llm)

    async def review(
        self,
        profile_doc: LearnerProfileDocument,
        session_id: str,
        interaction_store: InteractionStore,
    ) -> SessionReviewResult:
        """Analyse the session interaction log and return updated profile fields."""
        session_doc = await interaction_store.get_session(session_id)

        interaction_lines = []
        if session_doc:
            for i, ix in enumerate(session_doc.interactions, 1):
                quiz_info = ""
                if ix.quiz and ix.quiz.status == "submitted" and ix.student_response:
                    total = len(ix.quiz.questions)
                    score = ix.student_response.score
                    wrong = [
                        f"Q: \"{q.question[:60]}\" | correct={q.correct_answer} student={a.selected_option}"
                        for q, a in zip(ix.quiz.questions, ix.student_response.answers)
                        if a.selected_option.upper() != q.correct_answer.upper()
                    ]
                    quiz_info = f" | quiz={score}/{total}"
                    if wrong:
                        quiz_info += " | wrong: " + "; ".join(wrong)
                elif ix.quiz:
                    quiz_info = f" | quiz={ix.quiz.status}"

                interaction_lines.append(
                    f"{i}. Topic: {ix.meta.topic or 'unknown'} | "
                    f"Q: \"{ix.question[:80]}\"{quiz_info} | "
                    f"prompt: {ix.meta.difficulty or 'unknown'}"
                )

        interaction_block = "\n".join(interaction_lines) if interaction_lines else "No interactions recorded."

        def fmt(lst: list[str]) -> str:
            return ", ".join(lst) if lst else "none"

        def fmt_skills(skill_map: dict) -> str:
            strong = [k for k, v in skill_map.items() if v.score > 0]
            weak = [k for k, v in skill_map.items() if v.score <= 0]
            return f"strong={fmt(strong)}, weak={fmt(weak)}"

        prompt = SESSION_REVIEW_PROMPT.format(
            n=len(session_doc.interactions) if session_doc else 0,
            interaction_block=interaction_block,
            grade=profile_doc.grade,
            current_learning_styles=fmt(list(profile_doc.learning_style.keys())),
            current_strong=fmt([k for k, v in profile_doc.technical_skills.items() if v.score > 0]),
            current_weak=fmt([k for k, v in profile_doc.technical_skills.items() if v.score <= 0]),
            current_softskills_strong=fmt([k for k, v in profile_doc.softskills.items() if v.score > 0]),
            current_softskills_weak=fmt([k for k, v in profile_doc.softskills.items() if v.score <= 0]),
        )

        response = await self._llm.generate(
            system_prompt="You are a learning profile analyst for an educational AI system.",
            user_message=prompt,
        )

        result = self._parse_response(response)
        logger.info(
            "session_review.done",
            learner_id=profile_doc.learner_id,
            session_id=session_id,
            learning_styles=result.learning_styles,
            technically_strong=result.technically_strong,
        )
        return result

    def _parse_response(self, response: str) -> SessionReviewResult:
        parsed: dict[str, str] = {}
        for line in response.strip().split("\n"):
            if ": " in line:
                key, _, value = line.partition(": ")
                if not value:
                    key, _, value = line.partition(":")
                parsed[key.strip().upper()] = value.strip()

        def parse_list(raw: str) -> list[str]:
            if not raw or raw.lower() == "none":
                return []
            return [item.strip() for item in raw.split(",") if item.strip()]

        def parse_tags(raw: str, valid: set[str]) -> list[str]:
            return [t for t in parse_list(raw) if t in valid]

        return SessionReviewResult(
            technically_strong=parse_list(parsed.get("TECHNICALLY_STRONG", "")),
            technically_weak=parse_list(parsed.get("TECHNICALLY_WEAK", "")),
            softskills_strong=parse_tags(parsed.get("SOFTSKILLS_STRONG", ""), _VALID_SOFTSKILL_TAGS),
            softskills_weak=parse_tags(parsed.get("SOFTSKILLS_WEAK", ""), _VALID_SOFTSKILL_TAGS),
            learning_styles=parse_tags(parsed.get("LEARNING_STYLES", ""), _VALID_LEARNSTYLE_TAGS),
            governance_feedback=parsed.get("GOVERNANCE_FEEDBACK", ""),
        )

    def build_feedback_record(
        self,
        result: SessionReviewResult,
        session_id: str,
        user_id: str,
        grade: str,
        prompt_version: str = "",
    ) -> FeedbackRecord:
        return FeedbackRecord(
            user_id=user_id,
            session_id=session_id,
            prompt_version=prompt_version,
            architecture="sequential",
            grade=grade,
            verdict="session_review",
            user_signal="session_end",
            judge_cot_summary=result.governance_feedback,
            retry_count=0,
        )

    SOFTSKILL_TAGS = _VALID_SOFTSKILL_TAGS
    LEARNSTYLE_TAGS = _VALID_LEARNSTYLE_TAGS
