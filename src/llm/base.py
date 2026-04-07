"""Abstract base class for LLM clients."""

from abc import ABC, abstractmethod


class BaseLLMClient(ABC):
    """Unified interface for all LLM providers."""

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    async def generate(self, system_prompt: str, user_message: str) -> str:
        """Generate a completion given a system prompt and user message."""

    @abstractmethod
    async def generate_with_messages(self, messages: list[dict]) -> str:
        """Generate a completion from a full message list."""
