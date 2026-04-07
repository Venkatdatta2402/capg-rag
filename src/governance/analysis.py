"""Analysis Agent stub — pattern mining on accumulated feedback.

Loop B Step 1: Reads feedback records, separates prompt failures from
retrieval failures using the retrieval quality signal, and identifies
patterns (e.g. "v3_grade3_standard fails 40% of the time on unit
conversion topics when retrieval quality is GOOD").
"""

import structlog
from pydantic import BaseModel, Field

from src.models.feedback import FeedbackRecord

logger = structlog.get_logger()


class AnalysisResult(BaseModel):
    """Output of the Analysis Agent."""

    total_records: int = 0
    prompt_failure_rate: float = 0.0
    retrieval_failure_rate: float = 0.0
    weak_prompt_versions: list[str] = Field(default_factory=list)
    weak_topics: list[str] = Field(default_factory=list)
    recommendation: str = ""


class AnalysisAgent:
    """Mines patterns from feedback data to identify prompt improvement opportunities.

    Separates prompt failures (retrieval was GOOD but verdict was NOT_UNDERSTOOD)
    from retrieval failures (retrieval was POOR/MARGINAL) so that prompt changes
    are not made to fix retrieval-layer problems.
    """

    async def analyze(self, records: list[FeedbackRecord]) -> AnalysisResult:
        """Analyze feedback records for patterns.

        TODO: Implement full pattern mining with statistical significance tests.
        Current implementation provides basic failure rate computation.
        """
        if not records:
            return AnalysisResult(recommendation="No records to analyze.")

        total = len(records)
        prompt_failures = [
            r for r in records
            if r.verdict == "NOT_UNDERSTOOD" and r.retrieval_quality_flag == "GOOD"
        ]
        retrieval_failures = [
            r for r in records
            if r.retrieval_quality_flag in ("POOR", "MARGINAL")
        ]

        result = AnalysisResult(
            total_records=total,
            prompt_failure_rate=len(prompt_failures) / total if total else 0,
            retrieval_failure_rate=len(retrieval_failures) / total if total else 0,
            weak_prompt_versions=list({r.prompt_version for r in prompt_failures}),
            weak_topics=list({r.topic for r in prompt_failures}),
            recommendation="Prompt improvement needed" if len(prompt_failures) > total * 0.2 else "No action needed",
        )

        logger.info(
            "analysis.done",
            total=total,
            prompt_failures=len(prompt_failures),
            retrieval_failures=len(retrieval_failures),
        )
        return result
