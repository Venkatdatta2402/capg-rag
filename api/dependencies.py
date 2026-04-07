"""FastAPI dependency injection for shared services."""

from functools import lru_cache

from config.settings import settings
from src.llm.factory import get_llm_client
from src.orchestrator.base import BaseOrchestrator
from src.orchestrator.factory import get_orchestrator
from src.prompt_service.canary import CanaryRouter
from src.prompt_service.registry import PromptRegistry
from src.prompt_service.selector import PromptSelector
from src.storage.feedback_store import FeedbackStore
from src.storage.quiz_store import QuizStore
from src.storage.session_memory import SessionMemoryStore
from src.storage.user_profile import UserProfileStore


@lru_cache
def get_pipeline() -> BaseOrchestrator:
    """Return the active orchestrator pipeline (cached singleton)."""
    return get_orchestrator()


@lru_cache
def get_quiz_store() -> QuizStore:
    return QuizStore()


@lru_cache
def get_prompt_registry() -> PromptRegistry:
    return PromptRegistry()


@lru_cache
def get_feedback_store() -> FeedbackStore:
    return FeedbackStore()


@lru_cache
def get_profile_store() -> UserProfileStore:
    return UserProfileStore()


@lru_cache
def get_session_store() -> SessionMemoryStore:
    return SessionMemoryStore()


@lru_cache
def get_prompt_selector() -> PromptSelector:
    # Uses the context LLM for scoring teaching-style candidates against judge feedback.
    # The context model is lightweight (e.g. gpt-4o-mini) — appropriate for this task.
    llm = get_llm_client(settings.context_provider, settings.context_model)
    return PromptSelector(PromptRegistry(), CanaryRouter(), llm=llm)
