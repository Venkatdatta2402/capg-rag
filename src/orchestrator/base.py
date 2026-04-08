"""Base orchestrator interface."""

from abc import ABC, abstractmethod

from src.models.query import QueryInput
from src.models.response import GenerationResponse


class BaseOrchestrator(ABC):
    """Abstract pipeline orchestrator."""

    @abstractmethod
    async def run(self, query_input: QueryInput) -> GenerationResponse:
        """Execute the full pipeline for a single interaction."""
