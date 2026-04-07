"""Suggestion Agent stub — generates improved prompt versions.

Loop B Step 4: Uses analysis findings and risk assessment to generate
an improved, annotated prompt version that addresses identified weaknesses.
"""

import uuid

import structlog
from pydantic import BaseModel

from src.governance.analysis import AnalysisResult
from src.governance.risk import RiskAssessment
from src.models.prompt import PromptCandidate, PromptVersion

logger = structlog.get_logger()


class SuggestionResult(BaseModel):
    """Output of the Suggestion Agent."""

    candidate: PromptCandidate | None = None
    rationale: str = ""
    skipped: bool = False
    skip_reason: str = ""


class SuggestionAgent:
    """Generates improved prompt versions based on analysis and risk assessment.

    TODO: Implement LLM-powered prompt improvement:
    - Analyze failure patterns from Analysis Agent
    - Consider risk constraints from Risk Agent
    - Generate modified prompt addressing specific weaknesses
    - Annotate changes with rationale
    - Output a PromptCandidate for experiment testing
    """

    async def suggest(
        self,
        current_prompt: PromptVersion,
        analysis: AnalysisResult,
        risk: RiskAssessment,
    ) -> SuggestionResult:
        """Generate an improved prompt candidate.

        Current implementation: stub that explains what would happen.
        """
        if not risk.proceed:
            return SuggestionResult(
                skipped=True,
                skip_reason=f"Risk too high ({risk.risk_level}): {', '.join(risk.concerns)}",
            )

        if analysis.recommendation == "No action needed":
            return SuggestionResult(
                skipped=True,
                skip_reason="Analysis found no actionable improvement opportunities.",
            )

        candidate = PromptCandidate(
            candidate_id=f"candidate_{uuid.uuid4().hex[:8]}",
            parent_version_id=current_prompt.version_id,
            template=current_prompt.template,  # Placeholder — would be modified
            rationale=f"Addresses failures in topics: {', '.join(analysis.weak_topics)}",
            risk_score=risk.risk_score,
            status="pending",
        )

        logger.info(
            "suggestion.stub",
            candidate_id=candidate.candidate_id,
            parent=current_prompt.version_id,
            message="Suggestion generation not yet implemented — returning template copy",
        )

        return SuggestionResult(candidate=candidate, rationale=candidate.rationale)
