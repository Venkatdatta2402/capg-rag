"""Session Review Agent.

Runs asynchronously at the end of a session (fire-and-forget). Reads the full
Q&A history and session state, then outputs:
  - Updated technically_strong / technically_weak / softskills_strong / softskills_weak areas
  - Inferred or confirmed learning_style
  - Governance feedback (what worked / didn't) for the prompt governance pipeline
"""

import structlog

from src.agents.base import BaseAgent
from src.llm.base import BaseLLMClient
from src.models.learner import LearnerProfile, SessionState
from src.models.feedback import FeedbackRecord

logger = structlog.get_logger()


SESSION_REVIEW_PROMPT = """\
You are reviewing a student's learning session to update their long-term profile.

=== SUMMARY OF PAST INTERACTIONS (before last 5) ===
Covered topics  : {past_topics}
Common errors   : {past_errors}
Performance trend: {past_trend}

=== RECENT INTERACTIONS (last {n_recent}) ===
{recent_block}

Total retries across session: {total_retries}

=== CURRENT LEARNER PROFILE ===
Grade: {grade}
Learning styles: {current_learning_styles}
Technical strengths: {current_strong}
Technical weaknesses: {current_weak}
Soft skill strengths: {current_softskills_strong}
Soft skill weaknesses: {current_softskills_weak}

Based on the session data, provide updated profile fields.

For TECHNICALLY_STRONG and TECHNICALLY_WEAK use topic/concept names (free text).

For SOFTSKILLS_STRONG and SOFTSKILLS_WEAK choose only from this fixed tag set:
  Reasoning:         softskill:decomposition, softskill:abstraction, softskill:pattern_mapping
  Cognitive Control: softskill:working_memory, softskill:attention_control, softskill:process_discipline
  Metacognition:     softskill:error_detection, softskill:confidence_calibration, softskill:reflection

For LEARNING_STYLES choose ALL that apply from this fixed tag set (comma-separated):
  Input:       learnstyle:visual, learnstyle:textual, learnstyle:example_driven, learnstyle:abstract_first
  Interaction: learnstyle:guided, learnstyle:exploratory, learnstyle:step_by_step, learnstyle:challenge_based
  Feedback:    learnstyle:immediate_feedback, learnstyle:delayed_reflection, learnstyle:hint_sensitive

Respond in this exact format:
TECHNICALLY_STRONG: <comma-separated topics, or "none">
TECHNICALLY_WEAK: <comma-separated topics, or "none">
SOFTSKILLS_STRONG: <comma-separated tags from the softskill set above, or "none">
SOFTSKILLS_WEAK: <comma-separated tags from the softskill set above, or "none">
LEARNING_STYLES: <comma-separated tags from the learnstyle set above, or "none">
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
    """Parsed output of the session review."""

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
        self.softskills_strong = softskills_strong   # validated softskill: tags
        self.softskills_weak = softskills_weak       # validated softskill: tags
        self.learning_styles = learning_styles       # validated learnstyle: tags
        self.governance_feedback = governance_feedback


class SessionReviewAgent(BaseAgent):
    """Reviews the full session and produces an updated learner profile."""

    def __init__(self, llm: BaseLLMClient):
        super().__init__(llm)

    async def review(
        self,
        profile: LearnerProfile,
        session: SessionState,
    ) -> SessionReviewResult:
        """Analyse the session and return updated profile fields + governance feedback."""
        past = session.summary_of_past

        recent_lines = []
        for i, ix in enumerate(session.recent_interactions, 1):
            quiz_info = (
                f"quiz={ix.quiz_status}, score={ix.score}"
                if ix.quiz_status != "ignored"
                else "quiz=ignored"
            )
            recent_lines.append(
                f"{i}. Topic: {ix.topic or 'unknown'} | "
                f"Q: \"{ix.question[:80]}\" | "
                f"{quiz_info} | "
                f"retries={ix.retry_count} | prompt={ix.prompt_version}"
            )
        recent_block = "\n".join(recent_lines) if recent_lines else "No recent interactions."

        def fmt(lst: list[str]) -> str:
            return ", ".join(lst) if lst else "none"

        prompt = SESSION_REVIEW_PROMPT.format(
            past_topics=fmt(past.covered_topics),
            past_errors=fmt(past.common_errors),
            past_trend=past.performance_trend,
            n_recent=len(session.recent_interactions),
            recent_block=recent_block,
            total_retries=session.retry_count,
            grade=profile.grade,
            current_learning_styles=fmt(profile.learning_styles),
            current_strong=fmt(profile.technically_strong_areas),
            current_weak=fmt(profile.technically_weak_areas),
            current_softskills_strong=fmt(profile.softskills_strong_areas),
            current_softskills_weak=fmt(profile.softskills_weak_areas),
        )

        response = await self._llm.generate(
            system_prompt="You are a learning profile analyst for an educational AI system.",
            user_message=prompt,
        )

        result = self._parse_response(response)
        logger.info(
            "session_review.done",
            user_id=profile.user_id,
            session_id=session.session_id,
            learning_styles=result.learning_styles,
            technically_strong=result.technically_strong,
            technically_weak=result.technically_weak,
        )
        return result

    def _parse_response(self, response: str) -> SessionReviewResult:
        parsed: dict[str, str] = {}
        for line in response.strip().split("\n"):
            if ":" in line:
                # Split on first space-colon or first colon followed by a space
                # to handle keys like LEARNING_STYLES vs tag values like learnstyle:visual
                key, _, value = line.partition(": ")
                if not value:
                    # No space after colon — still parse but key may be malformed
                    key, _, value = line.partition(":")
                parsed[key.strip().upper()] = value.strip()

        def parse_list(raw: str) -> list[str]:
            if not raw or raw.lower() == "none":
                return []
            return [item.strip() for item in raw.split(",") if item.strip()]

        def parse_tags(raw: str, valid: set[str]) -> list[str]:
            return [t for t in parse_list(raw) if t in valid]

        # LEARNING_STYLES key has a colon in value tags, so partition on first colon only
        # The response parser already does that via line.partition(":") — however the value
        # for LEARNING_STYLES will be e.g. "learnstyle:visual, learnstyle:step_by_step".
        # We stored the full raw value after the first colon, so re-join for multi-tag values.
        raw_styles = parsed.get("LEARNING_STYLES", "")

        return SessionReviewResult(
            technically_strong=parse_list(parsed.get("TECHNICALLY_STRONG", "")),
            technically_weak=parse_list(parsed.get("TECHNICALLY_WEAK", "")),
            softskills_strong=parse_tags(parsed.get("SOFTSKILLS_STRONG", ""), _VALID_SOFTSKILL_TAGS),
            softskills_weak=parse_tags(parsed.get("SOFTSKILLS_WEAK", ""), _VALID_SOFTSKILL_TAGS),
            learning_styles=parse_tags(raw_styles, _VALID_LEARNSTYLE_TAGS),
            governance_feedback=parsed.get("GOVERNANCE_FEEDBACK", ""),
        )

    def build_feedback_record(
        self,
        result: SessionReviewResult,
        session: SessionState,
        grade: str,
    ) -> FeedbackRecord:
        """Package governance feedback as a FeedbackRecord for the governance pipeline."""
        prompt_versions = list({ix.prompt_version for ix in session.recent_interactions if ix.prompt_version})
        return FeedbackRecord(
            user_id=session.user_id,
            session_id=session.session_id,
            prompt_version=prompt_versions[0] if prompt_versions else "",
            architecture="sequential",
            grade=grade,
            verdict="session_review",
            user_signal="session_end",
            judge_cot_summary=result.governance_feedback,
            retry_count=session.retry_count,
        )

    # ------------------------------------------------------------------
    # Tag taxonomy reference (for prompt authors and tests)
    # ------------------------------------------------------------------

    SOFTSKILL_TAGS = _VALID_SOFTSKILL_TAGS
    LEARNSTYLE_TAGS = _VALID_LEARNSTYLE_TAGS
