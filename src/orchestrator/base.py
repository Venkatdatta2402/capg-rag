"""Base orchestrator interface for both architectures."""

from abc import ABC, abstractmethod

from src.models.query import QueryInput
from src.models.response import GenerationResponse


class BaseOrchestrator(ABC):
    """Abstract pipeline orchestrator.

    Both Architecture A and B implement this interface, allowing
    the API layer to call orchestrator.run() regardless of which
    architecture is active.
    """

    @abstractmethod
    async def run(self, query_input: QueryInput) -> GenerationResponse:
        """Execute the full runtime pipeline for a single interaction.

        Args:
            query_input: Raw learner input.

        Returns:
            GenerationResponse with answer and metadata.
        """
