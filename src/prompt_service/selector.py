"""Context-driven prompt selection.

Flow per turn:
  1. LLM analyzes LearnerProfileDocument + current query → reasons all context signals
  2. ContextObject is built from LLM output + passed-in retry state
  3. ContextObject tags are matched against prompt registry → best prompt selected
  4. Returns (PromptVersion, ContextObject) — caller saves ContextObject to session
"""

import structlog

from src.llm.base import BaseLLMClient
from src.models.profile_document import LearnerProfileDocument
from src.models.query import ContextObject, EnrichedQuery
from src.models.prompt import PromptVersion
from src.prompt_service.registry import PromptRegistry
from src.prompt_service.canary import CanaryRouter

logger = structlog.get_logger()

_CONTEXT_ANALYSIS_PROMPT = """\
You are a learning context analyzer for a CBSE/NCERT educational AI.

Given the learner's long-term profile and their current query, identify the most
relevant learning context signals FOR THIS SPECIFIC INTERACTION.

=== LEARNER PROFILE ===
Grade: {grade}

Technical skills (score > 0 = relative strength, score ≤ 0 = relative weakness):
{technical_skills_block}

Soft skills:
{softskills_block}

Learning style preferences:
{learning_style_block}

=== CURRENT QUERY ===
Topic   : {topic}
Query   : {query}
Type    : {query_type}

=== TASK ===
Reason about which signals are most relevant to THIS query and this learner right now.
Consider: does the query topic align with their technical strengths or weaknesses?
Which soft skills will help or hinder them on THIS specific task?
Which learning styles suit how this topic is best learned?

Choose ONLY from these fixed tag sets:

LEARNING_STYLES (pick 1-3 most relevant):
  learnstyle:visual, learnstyle:textual, learnstyle:example_driven, learnstyle:abstract_first,
  learnstyle:guided, learnstyle:exploratory, learnstyle:step_by_step, learnstyle:challenge_based,
  learnstyle:immediate_feedback, learnstyle:delayed_reflection, learnstyle:hint_sensitive

SOFTSKILLS (pick 0-3 each — only tags that are clearly relevant to this query):
  softskill:decomposition, softskill:abstraction, softskill:pattern_mapping,
  softskill:working_memory, softskill:attention_control, softskill:process_discipline,
  softskill:error_detection, softskill:confidence_calibration, softskill:reflection

Respond in this exact format:
LEARNING_STYLES: <comma-separated learnstyle: tags, or "none">
SOFTSKILLS_STRONG: <comma-separated softskill: tags from their strengths relevant here, or "none">
SOFTSKILLS_WEAK: <comma-separated softskill: tags from their weaknesses relevant here, or "none">
TOPIC_STRENGTH: topic:strong | topic:weak
"""

_VALID_LEARNSTYLE = {
    "learnstyle:visual", "learnstyle:textual", "learnstyle:example_driven",
    "learnstyle:abstract_first", "learnstyle:guided", "learnstyle:exploratory",
    "learnstyle:step_by_step", "learnstyle:challenge_based", "learnstyle:immediate_feedback",
    "learnstyle:delayed_reflection", "learnstyle:hint_sensitive",
}
_VALID_SOFTSKILL = {
    "softskill:decomposition", "softskill:abstraction", "softskill:pattern_mapping",
    "softskill:working_memory", "softskill:attention_control", "softskill:process_discipline",
    "softskill:error_detection", "softskill:confidence_calibration", "softskill:reflection",
}


class PromptSelector:
    """Selects the appropriate prompt version based on learner profile + query context."""

    def __init__(
        self,
        registry: PromptRegistry,
        canary_router: CanaryRouter,
        llm: BaseLLMClient | None = None,
    ):
        self._registry = registry
        self._canary = canary_router
        self._llm = llm

    async def select(
        self,
        profile_doc: LearnerProfileDocument,
        enriched: EnrichedQuery,
        retry_mode: bool = False,
        retry_count: int = 0,
    ) -> tuple[PromptVersion, ContextObject]:
        """Analyze profile + query with LLM, build ContextObject, pick best prompt."""
        context_obj = await self._analyze_context(profile_doc, enriched, retry_mode, retry_count)
        context_tags = self._build_context_tags(context_obj)

        active = await self._registry.list_active(grade=profile_doc.grade)

        if not active:
            logger.warning("prompt_selector.no_active_prompts", grade=profile_doc.grade)
            fallback = self._best_fallback(profile_doc.grade, context_tags)
            return fallback, context_obj

        control = self._best_tag_match(active, context_tags)
        candidates = await self._registry.list_candidates()

        if candidates:
            selected = self._canary.route(control, candidates[0])
        else:
            selected = control

        logger.info(
            "prompt_selector.selected",
            version_id=selected.version_id,
            context_tags=list(context_tags),
            topic_strength=context_obj.topic_strength,
        )
        return selected, context_obj

    async def _analyze_context(
        self,
        profile_doc: LearnerProfileDocument,
        enriched: EnrichedQuery,
        retry_mode: bool,
        retry_count: int,
    ) -> ContextObject:
        if not self._llm:
            return ContextObject(
                grade=profile_doc.grade,
                retry_mode=retry_mode,
                retry_count=retry_count,
            )

        def fmt_skills(skill_map: dict) -> str:
            if not skill_map:
                return "  (no data yet)"
            return "\n".join(
                f"  {k}: score={v.score:.2f} ({v.count} obs)"
                for k, v in sorted(skill_map.items(), key=lambda x: -abs(x[1].score))
            )

        prompt = _CONTEXT_ANALYSIS_PROMPT.format(
            grade=profile_doc.grade or "unknown",
            technical_skills_block=fmt_skills(profile_doc.technical_skills),
            softskills_block=fmt_skills(profile_doc.softskills),
            learning_style_block=fmt_skills(profile_doc.learning_style),
            topic=enriched.topic or "unknown",
            query=enriched.original_text[:200],
            query_type=enriched.query_type or "unknown",
        )

        try:
            raw = await self._llm.generate(
                system_prompt="You are a learning context analyzer for an educational AI system.",
                user_message=prompt,
            )
            learning_styles, ss_strong, ss_weak, topic_strength = self._parse_context(raw)
        except Exception as exc:
            logger.warning("prompt_selector.context_analysis_failed", error=str(exc))
            learning_styles, ss_strong, ss_weak, topic_strength = [], [], [], "topic:weak"

        return ContextObject(
            grade=profile_doc.grade,
            learning_styles=learning_styles,
            softskills_strong=ss_strong,
            softskills_weak=ss_weak,
            topic_strength=topic_strength,
            retry_mode=retry_mode,
            retry_count=retry_count,
        )

    def _parse_context(self, raw: str) -> tuple[list[str], list[str], list[str], str]:
        parsed: dict[str, str] = {}
        for line in raw.strip().split("\n"):
            if ": " in line:
                key, _, value = line.partition(": ")
                parsed[key.strip().upper()] = value.strip()

        def parse_tags(raw_val: str, valid: set[str]) -> list[str]:
            if not raw_val or raw_val.lower() == "none":
                return []
            return [t.strip() for t in raw_val.split(",") if t.strip() in valid]

        topic_raw = parsed.get("TOPIC_STRENGTH", "topic:weak").lower().strip()
        topic_strength = topic_raw if topic_raw in ("topic:strong", "topic:weak") else "topic:weak"

        return (
            parse_tags(parsed.get("LEARNING_STYLES", ""), _VALID_LEARNSTYLE),
            parse_tags(parsed.get("SOFTSKILLS_STRONG", ""), _VALID_SOFTSKILL),
            parse_tags(parsed.get("SOFTSKILLS_WEAK", ""), _VALID_SOFTSKILL),
            topic_strength,
        )

    @staticmethod
    def _build_context_tags(context: ContextObject) -> set[str]:
        tags: set[str] = set()
        tags.update(context.learning_styles)
        tags.add(context.topic_strength)
        tags.update(context.softskills_strong)
        tags.update(context.softskills_weak)
        if context.retry_mode:
            tags.add("retry")
        return tags

    @staticmethod
    def _best_tag_match(prompts: list[PromptVersion], context_tags: set[str]) -> PromptVersion:
        return max(prompts, key=lambda p: len(set(p.tags) & context_tags))

    @classmethod
    def _best_fallback(cls, grade: str, context_tags: set[str]) -> PromptVersion:
        fallbacks = cls._builtin_fallbacks(grade)
        return cls._best_tag_match(fallbacks, context_tags)

    @staticmethod
    def _builtin_fallbacks(grade: str) -> list[PromptVersion]:
        return [
            PromptVersion(
                version_id="builtin_standard",
                template=(
                    "You are a clear and patient CBSE/NCERT tutor. "
                    "Explain concepts step-by-step with examples from daily life."
                ),
                grade=grade,
                variant="standard",
                tags=["learnstyle:example_driven", "learnstyle:step_by_step", "topic:weak"],
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
                tags=[
                    "learnstyle:guided", "learnstyle:step_by_step", "learnstyle:hint_sensitive",
                    "topic:weak", "softskill:attention_control", "retry",
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
                tags=[
                    "learnstyle:abstract_first", "learnstyle:challenge_based",
                    "topic:strong", "softskill:abstraction", "softskill:pattern_mapping",
                ],
            ),
        ]
