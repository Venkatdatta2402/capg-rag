"""FastAPI dependency injection for shared services."""

from functools import lru_cache

from config.settings import settings
from src.agents.session_review import SessionReviewAgent
from src.llm.factory import get_llm_client
from src.orchestrator.pipeline import Pipeline
from src.prompt_service.canary import CanaryRouter
from src.prompt_service.registry import PromptRegistry
from src.prompt_service.selector import PromptSelector
from src.storage.feedback_store import FeedbackStore
from src.storage.interaction_store import InteractionStore
from src.storage.session_memory import SessionMemoryStore
from src.storage.user_profile import UserProfileStore


@lru_cache
def get_pipeline() -> Pipeline:
    """Return the active pipeline (cached singleton)."""
    return Pipeline()


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
    llm = get_llm_client(settings.context_provider, settings.context_model)
    return PromptSelector(PromptRegistry(), CanaryRouter(), llm=llm)


@lru_cache
def get_session_review_agent() -> SessionReviewAgent:
    llm = get_llm_client(settings.judge_provider, settings.judge_model)
    return SessionReviewAgent(llm)


@lru_cache
def get_interaction_store() -> InteractionStore:
    return InteractionStore()
