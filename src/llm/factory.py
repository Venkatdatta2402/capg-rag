"""Factory for creating LLM clients by provider and model."""

from src.llm.base import BaseLLMClient
from src.llm.openai_client import OpenAIClient
from src.llm.gemini_client import GeminiClient

_PROVIDERS = {
    "openai": OpenAIClient,
    "gemini": GeminiClient,
}


def get_llm_client(provider: str, model: str) -> BaseLLMClient:
    """Create an LLM client for the given provider and model.

    Args:
        provider: One of "openai", "gemini".
        model: Model identifier (e.g. "gpt-4.1", "gemini-pro").

    Returns:
        An instance of BaseLLMClient.
    """
    cls = _PROVIDERS.get(provider)
    if cls is None:
        raise ValueError(f"Unknown LLM provider '{provider}'. Options: {list(_PROVIDERS)}")
    return cls(model)
