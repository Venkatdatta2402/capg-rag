"""Context-driven prompt selection — picks the best versioned prompt for the current learner."""

import structlog

from src.llm.base import BaseLLMClient
from src.models.query import ContextObject
from src.models.prompt import PromptVersion
from src.prompt_service.registry import PromptRegistry
from src.prompt_service.canary import CanaryRouter

logger = structlog.get_logger()

# Prompt shown to the LLM to score teaching-style candidates against judge feedback.
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
    """Selects the appropriate prompt version based on the context object."""

    def __init__(
        self,
        registry: PromptRegistry,
        canary_router: CanaryRouter,
        llm: BaseLLMClient | None = None,
    ):
        self._registry = registry
        self._canary = canary_router
        self._llm = llm          # used only by select_candidates_from_feedback()

    async def select(self, context: ContextObject) -> PromptVersion:
        """Select the best prompt version for the given context.

        Selection rules:
        - retry_mode=True → remedial variant
        - comprehension_level="high" → advanced variant
        - otherwise → standard variant

        If a canary candidate exists, the canary router decides whether to
        serve the candidate or the control version.
        """
        if context.retry_mode:
            variant = "remedial"
        elif context.comprehension_level == "high":
            variant = "advanced"
        else:
            variant = "standard"

        active = await self._registry.list_active(
            grade=context.learner_grade,
            variant=variant,
        )

        if not active:
            logger.warning("prompt_selector.no_match", grade=context.learner_grade, variant=variant)
            return PromptVersion(
                version_id="default",
                template="You are a helpful teaching assistant. Answer the student's question clearly.",
                grade=context.learner_grade,
                variant=variant,
            )

        control = active[0]

        candidates = await self._registry.list_candidates()
        if candidates:
            selected = self._canary.route(control, candidates[0])
            logger.info(
                "prompt_selector.selected",
                version_id=selected.version_id,
                cohort="canary" if selected.version_id != control.version_id else "control",
            )
            return selected

        logger.info("prompt_selector.selected", version_id=control.version_id, cohort="control")
        return control

    async def select_candidates_from_feedback(
        self,
        grade: str,
        retrieval_feedback: str,
        n: int = 3,
    ) -> list[PromptVersion]:
        """Return the n most suitable teaching-style prompts for the identified gaps.

        The judge's retrieval_feedback describes what concepts were missing from
        the learner's understanding. This method uses an LLM to score every active
        prompt (by its description + tags) against that feedback, then returns the
        top n — one per distinct variant so the MCQ choices are always meaningfully
        different teaching styles, not three flavours of the same approach.

        Falls back to one-per-variant selection when no LLM is configured or the
        registry is empty (safe for unit tests and early-stage deployments).
        """
        active = await self._registry.list_active(grade=grade)

        if not active:
            logger.warning("prompt_selector.no_active_prompts", grade=grade)
            return self._builtin_fallbacks(grade, n)

        # Without an LLM, fall back to one-per-variant (no feedback scoring)
        if not self._llm or not retrieval_feedback.strip():
            logger.info("prompt_selector.feedback_scoring_skipped", reason="no llm or empty feedback")
            return self._one_per_variant(active, n)

        # Build the candidates block for the scoring prompt
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

        # Rank by score descending; deduplicate by variant so MCQ choices are diverse
        ranked = sorted(zip(scores, active), key=lambda x: -x[0])
        seen_variants: set[str] = set()
        chosen: list[PromptVersion] = []
        for _, version in ranked:
            if version.variant not in seen_variants:
                chosen.append(version)
                seen_variants.add(version.variant)
            if len(chosen) == n:
                break

        # Pad with fallbacks if fewer than n distinct variants exist in registry
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

    @staticmethod
    def _parse_scores(raw: str, expected: int) -> list[float]:
        """Parse SCORE_N: <value> lines into a float list."""
        scores: list[float] = []
        for line in raw.strip().split("\n"):
            if line.upper().startswith("SCORE_"):
                try:
                    scores.append(float(line.split(":")[-1].strip()))
                except ValueError:
                    scores.append(0.0)
        # Pad or trim to expected length
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
                tags=["step-by-step", "conceptual", "examples"],
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
                tags=["analogy", "scaffolded", "remedial", "slow-paced"],
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
                tags=["application", "cross-chapter", "advanced", "higher-order"],
            ),
        ]
        return fallbacks[:n]
