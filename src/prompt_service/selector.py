"""Context-driven prompt selection — picks the best versioned prompt for the current learner."""

import structlog

from src.llm.base import BaseLLMClient
from src.models.query import ContextObject
from src.models.prompt import PromptVersion
from src.prompt_service.registry import PromptRegistry
from src.prompt_service.canary import CanaryRouter

logger = structlog.get_logger()

# ── LLM prompt: topic strength estimation ─────────────────────────────────────

_TOPIC_STRENGTH_PROMPT = """\
You are assessing a learner's likely proficiency in a specific topic based on their long-term profile.

Topic being queried: {topic}

Learner's technically strong areas: {strong_areas}
Learner's technically weak areas: {weak_areas}

Consider semantic proximity — e.g. "multiplication" as a strong area indicates solid preparation
for "algebra basics"; "fractions" as a weak area suggests difficulty with "ratio and proportion".
If the profile provides no clear signal, lean toward WEAK (it is safer to over-support).

Respond with exactly one of these tags (no explanation):
topic:strong   — the topic aligns clearly with the learner's strengths
topic:weak     — the topic aligns with known weaknesses, adjacent gaps, or is simply unknown
"""

# ── LLM prompt: teaching-style candidate scoring against judge feedback ────────

_SCORING_PROMPT = """\
A student quiz was evaluated and the following learning gaps were identified:

{retrieval_feedback}

Below are {n} teaching approaches available in the system. Score each from 0 to 10
based on how well it would address the identified gaps. A score of 10 means the
approach directly targets the type of misunderstanding described; 0 means it is
irrelevant or would not help.

{candidates_block}

Rules:
- Scores must be integers 0–10.
- Scores do not need to be unique.
- Do not explain your reasoning.

Respond in this exact format (one line per candidate, same order):
SCORE_1: <integer>
SCORE_2: <integer>
...
SCORE_{n}: <integer>
"""


class PromptSelector:
    """Selects the appropriate prompt version based on learner context and query topic."""

    def __init__(
        self,
        registry: PromptRegistry,
        canary_router: CanaryRouter,
        llm: BaseLLMClient | None = None,
    ):
        self._registry = registry
        self._canary = canary_router
        self._llm = llm

    async def select(self, context: ContextObject, topic: str) -> PromptVersion:
        """Select the best prompt by matching context signals to prompt tags.

        Context signals → tag set:
          learning_style       → e.g. "example-driven"
          topic strength (LLM) → "topic-strong" | "topic-weak"
          softskills_strong    → "softskill-strong"  (if any)
          softskills_weak      → "softskill-weak"    (if any)
          retry_mode           → "retry"             (if True)

        Each active prompt is scored by how many of its tags appear in the
        context tag set. The highest-scoring prompt wins; ties broken by
        registry order. Falls back to builtins if registry is empty.
        """
        topic_strength = await self._estimate_topic_strength(
            topic=topic,
            strong_areas=context.technically_strong_areas,
            weak_areas=context.technically_weak_areas,
        )

        context_tags = self._build_context_tags(context, topic_strength)

        active = await self._registry.list_active(grade=context.learner_grade)

        if not active:
            logger.warning("prompt_selector.no_active_prompts", grade=context.learner_grade)
            fallback = self._best_fallback(context.learner_grade, context_tags)
            return fallback

        control = self._best_tag_match(active, context_tags)

        candidates = await self._registry.list_candidates()
        if candidates:
            selected = self._canary.route(control, candidates[0])
            logger.info(
                "prompt_selector.selected",
                version_id=selected.version_id,
                context_tags=list(context_tags),
                topic_strength=topic_strength,
                cohort="canary" if selected.version_id != control.version_id else "control",
            )
            return selected

        logger.info(
            "prompt_selector.selected",
            version_id=control.version_id,
            context_tags=list(context_tags),
            topic_strength=topic_strength,
            cohort="control",
        )
        return control

    # ------------------------------------------------------------------
    # Tag set construction and matching
    # ------------------------------------------------------------------

    @staticmethod
    def _build_context_tags(context: ContextObject, topic_strength: str) -> set[str]:
        """Convert context signals into a set of tags to match against prompt tags.

        Tags added:
          - All learnstyle: tags from the learner's learning_styles list
          - topic:strong or topic:weak from LLM estimate
          - All softskill: tags from both strong and weak areas
          - retry if retry_mode is True
        """
        tags: set[str] = set()
        tags.update(context.learning_styles)
        tags.add(topic_strength)  # already "topic:strong" or "topic:weak"
        tags.update(context.softskills_strong_areas)
        tags.update(context.softskills_weak_areas)
        if context.retry_mode:
            tags.add("retry")
        return tags

    @staticmethod
    def _best_tag_match(prompts: list[PromptVersion], context_tags: set[str]) -> PromptVersion:
        """Return the prompt whose tags have the most overlap with context_tags."""
        return max(prompts, key=lambda p: len(set(p.tags) & context_tags))

    # ------------------------------------------------------------------
    # LLM: topic strength estimation
    # ------------------------------------------------------------------

    async def _estimate_topic_strength(
        self,
        topic: str,
        strong_areas: list[str],
        weak_areas: list[str],
    ) -> str:
        """Ask the LLM whether this topic is STRONG or WEAK for the learner.

        Falls back to WEAK (safer/more supportive) when the LLM is unavailable.
        """
        if not self._llm or not topic:
            return "topic:weak"

        def fmt(lst: list[str]) -> str:
            return ", ".join(lst) if lst else "none"

        prompt = _TOPIC_STRENGTH_PROMPT.format(
            topic=topic,
            strong_areas=fmt(strong_areas),
            weak_areas=fmt(weak_areas),
        )

        try:
            raw = await self._llm.generate(
                system_prompt="You are a learner proficiency assessor for an educational AI system.",
                user_message=prompt,
            )
            strength = raw.strip().lower().split()[0]
            if strength not in ("topic:strong", "topic:weak"):
                logger.warning("prompt_selector.topic_strength_unexpected", raw=raw)
                return "topic:weak"
            return strength
        except Exception as exc:
            logger.warning("prompt_selector.topic_strength_failed", error=str(exc))
            return "topic:weak"

    # ------------------------------------------------------------------
    # Feedback-aware candidate selection (used by /quiz/submit on FAILED)
    # ------------------------------------------------------------------

    async def select_candidates_from_feedback(
        self,
        grade: str,
        retrieval_feedback: str,
        n: int = 3,
    ) -> list[PromptVersion]:
        """Return the n most suitable teaching-style prompts for the identified gaps.

        Uses an LLM to score every active prompt against the judge's retrieval_feedback,
        returning the top n — one per distinct variant for meaningful MCQ diversity.
        Falls back to one-per-variant when no LLM is configured.
        """
        active = await self._registry.list_active(grade=grade)

        if not active:
            logger.warning("prompt_selector.no_active_prompts", grade=grade)
            return self._builtin_fallbacks(grade, n)

        if not self._llm or not retrieval_feedback.strip():
            logger.info("prompt_selector.feedback_scoring_skipped", reason="no llm or empty feedback")
            return self._one_per_variant(active, n)

        candidates_block = "\n".join(
            f"Candidate {i + 1}: {v.description or v.variant.capitalize()} "
            f"[tags: {', '.join(v.tags) or 'none'}]"
            for i, v in enumerate(active)
        )

        prompt = _SCORING_PROMPT.format(
            retrieval_feedback=retrieval_feedback,
            n=len(active),
            candidates_block=candidates_block,
        )

        try:
            raw = await self._llm.generate(
                system_prompt="You are a teaching-strategy evaluator for an educational AI system.",
                user_message=prompt,
            )
            scores = self._parse_scores(raw, len(active))
        except Exception as exc:
            logger.warning("prompt_selector.scoring_failed", error=str(exc))
            return self._one_per_variant(active, n)

        ranked = sorted(zip(scores, active), key=lambda x: -x[0])
        seen_variants: set[str] = set()
        chosen: list[PromptVersion] = []
        for _, version in ranked:
            if version.variant not in seen_variants:
                chosen.append(version)
                seen_variants.add(version.variant)
            if len(chosen) == n:
                break

        if len(chosen) < n:
            for fb in self._builtin_fallbacks(grade, n):
                if fb.variant not in seen_variants:
                    chosen.append(fb)
                    seen_variants.add(fb.variant)
                if len(chosen) == n:
                    break

        logger.info(
            "prompt_selector.feedback_candidates_selected",
            grade=grade,
            count=len(chosen),
            versions=[v.version_id for v in chosen],
        )
        return chosen

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @classmethod
    def _best_fallback(cls, grade: str, context_tags: set[str]) -> PromptVersion:
        """Pick the best-matching builtin when the registry is empty."""
        fallbacks = cls._builtin_fallbacks(grade, n=3)
        return cls._best_tag_match(fallbacks, context_tags)

    @staticmethod
    def _parse_scores(raw: str, expected: int) -> list[float]:
        scores: list[float] = []
        for line in raw.strip().split("\n"):
            if line.upper().startswith("SCORE_"):
                try:
                    scores.append(float(line.split(":")[-1].strip()))
                except ValueError:
                    scores.append(0.0)
        while len(scores) < expected:
            scores.append(0.0)
        return scores[:expected]

    @staticmethod
    def _one_per_variant(versions: list[PromptVersion], n: int) -> list[PromptVersion]:
        seen: set[str] = set()
        result: list[PromptVersion] = []
        for v in versions:
            if v.variant not in seen:
                result.append(v)
                seen.add(v.variant)
            if len(result) == n:
                break
        return result

    @staticmethod
    def _builtin_fallbacks(grade: str, n: int) -> list[PromptVersion]:
        fallbacks = [
            PromptVersion(
                version_id="builtin_standard",
                template=(
                    "You are a clear and patient CBSE/NCERT tutor. "
                    "Explain concepts step-by-step with examples from daily life."
                ),
                grade=grade,
                variant="standard",
                description="Step-by-step explanation with everyday examples",
                tags=[
                    "learnstyle:example_driven", "learnstyle:step_by_step",
                    "learnstyle:immediate_feedback",
                    "topic:weak",
                ],
            ),
            PromptVersion(
                version_id="builtin_remedial",
                template=(
                    "You are a supportive CBSE/NCERT tutor. The learner is struggling — "
                    "break the concept into the smallest possible steps, "
                    "use analogies, and check understanding at each step."
                ),
                grade=grade,
                variant="remedial",
                description="Ultra-simplified breakdown with analogies for struggling learners",
                tags=[
                    "learnstyle:guided", "learnstyle:step_by_step", "learnstyle:hint_sensitive",
                    "learnstyle:immediate_feedback",
                    "topic:weak",
                    "softskill:attention_control", "softskill:working_memory",
                    "retry",
                ],
            ),
            PromptVersion(
                version_id="builtin_advanced",
                template=(
                    "You are a rigorous CBSE/NCERT tutor. Challenge the learner "
                    "with deeper connections across chapters, precise terminology, "
                    "and application-level thinking embedded in your explanation."
                ),
                grade=grade,
                variant="advanced",
                description="Rigorous explanation with cross-chapter connections and application",
                tags=[
                    "learnstyle:abstract_first", "learnstyle:challenge_based",
                    "learnstyle:exploratory", "learnstyle:delayed_reflection",
                    "topic:strong",
                    "softskill:abstraction", "softskill:pattern_mapping",
                ],
            ),
        ]
        return fallbacks[:n]
