"""Risk Agent stub — predicts regression risk for prompt changes.

Loop B Step 2: Assesses whether a proposed prompt change could cause
regression, considering complexity, provider compatibility, and
historical patterns.
"""

import structlog
from pydantic import BaseModel

from src.governance.analysis import AnalysisResult
from src.models.prompt import PromptVersion

logger = structlog.get_logger()


class RiskAssessment(BaseModel):
    """Output of the Risk Agent."""

    risk_score: float = 0.0        # 0.0 (safe) to 1.0 (high risk)
    risk_level: str = "low"        # low, medium, high
    concerns: list[str] = []
    proceed: bool = True


class RiskAgent:
    """Predicts regression risk before prompt changes are tested.

    TODO: Implement full risk model considering:
    - Historical performance of similar changes
    - Complexity of the new prompt vs the old
    - Provider-specific behavior differences
    - Affected learner population size
    """

    async def assess(
        self,
        current_prompt: PromptVersion,
        analysis: AnalysisResult,
    ) -> RiskAssessment:
        """Assess risk of modifying the given prompt based on analysis findings.

        Current implementation: basic heuristic stub.
        """
        concerns = []

        if analysis.prompt_failure_rate > 0.4:
            concerns.append("High failure rate suggests significant prompt issues")
        if len(analysis.weak_topics) > 3:
            concerns.append("Failures span multiple topics — broad change needed")

        risk_score = min(analysis.prompt_failure_rate * 1.5, 1.0)
        risk_level = "high" if risk_score > 0.6 else "medium" if risk_score > 0.3 else "low"

        result = RiskAssessment(
            risk_score=round(risk_score, 2),
            risk_level=risk_level,
            concerns=concerns,
            proceed=risk_score < 0.8,
        )

        logger.info("risk.assessed", risk_level=risk_level, risk_score=result.risk_score)
        return result
