"""Canary deployment routing — sends a small percentage of traffic to candidate prompts."""

import random

import structlog

from src.models.prompt import PromptCandidate, PromptVersion
from config.settings import settings

logger = structlog.get_logger()


class CanaryRouter:
    """Routes traffic between control and canary prompt versions."""

    def __init__(self, canary_percent: int | None = None):
        self._canary_percent = canary_percent or settings.canary_traffic_percent

    def route(self, control: PromptVersion, candidate: PromptCandidate) -> PromptVersion:
        """Decide whether to serve the control or canary version.

        Returns the control version for most traffic and the candidate
        for a small percentage (default 5%).
        """
        if random.randint(1, 100) <= self._canary_percent:
            logger.info("canary_router.serving_canary", candidate_id=candidate.candidate_id)
            return PromptVersion(
                version_id=candidate.candidate_id,
                template=candidate.template,
                grade=control.grade,
                variant=control.variant,
                status="candidate",
            )
        return control
