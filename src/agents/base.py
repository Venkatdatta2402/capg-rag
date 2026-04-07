"""Base agent interface for all agents in the system."""

from abc import ABC, abstractmethod

from src.llm.base import BaseLLMClient


class BaseAgent(ABC):
    """Abstract base for all agents (context, judge, RAG, etc.)."""

    def __init__(self, llm: BaseLLMClient):
        self._llm = llm

    @abstractmethod
    async def run(self, **kwargs):
        """Execute the agent's primary task."""
