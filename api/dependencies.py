"""FastAPI dependency injection for shared services."""

from functools import lru_cache

from config.settings import settings
from src.agents.eval_agent import EvalAgent
from src.agents.session_review import SessionReviewAgent
from src.llm.factory import get_llm_client
from src.orchestrator.pipeline import Pipeline
from src.prompt_service.registry import PromptRegistry
from src.storage.eval_store import EvalStore
from src.storage.feedback_store import FeedbackStore
from src.storage.interaction_store import InteractionStore
from src.storage.learner_profile_store import LearnerProfileStore
from src.storage.session_memory import SessionMemoryStore


@lru_cache
def get_pipeline() -> Pipeline:
    return Pipeline()


@lru_cache
def get_prompt_registry() -> PromptRegistry:
    return PromptRegistry()


@lru_cache
def get_feedback_store() -> FeedbackStore:
    return FeedbackStore()


@lru_cache
def get_learner_profile_store() -> LearnerProfileStore:
    return LearnerProfileStore()


@lru_cache
def get_session_store() -> SessionMemoryStore:
    return SessionMemoryStore()


@lru_cache
def get_session_review_agent() -> SessionReviewAgent:
    llm = get_llm_client(settings.judge_provider, settings.judge_model)
    return SessionReviewAgent(llm)


@lru_cache
def get_interaction_store() -> InteractionStore:
    return InteractionStore()


@lru_cache
def get_eval_store() -> EvalStore:
    return EvalStore()


@lru_cache
def get_eval_agent() -> EvalAgent:
    llm = get_llm_client(settings.judge_provider, settings.judge_model)
    return EvalAgent(llm)
