"""Shared test fixtures."""

import pytest

from src.models.query import QueryInput, EnrichedQuery, ContextObject
from src.models.learner import LearnerProfile, SessionState
from src.models.retrieval import RetrievalResult, RerankedChunk
from src.models.prompt import PromptVersion


@pytest.fixture
def sample_query_input():
    return QueryInput(
        query_text="what is M to cm?",
        user_id="test_user_1",
        session_id="test_session_1",
    )


@pytest.fixture
def sample_profile():
    return LearnerProfile(
        user_id="test_user_1",
        grade="Class 3",
        board="CBSE",
        learning_styles=["learnstyle:example_driven", "learnstyle:step_by_step", "learnstyle:immediate_feedback"],
        technically_strong_areas=["division", "multiplication"],
        technically_weak_areas=["unit conversion"],
        softskills_strong_areas=["softskill:pattern_mapping"],
        softskills_weak_areas=["softskill:attention_control"],
    )


@pytest.fixture
def sample_session():
    return SessionState(
        session_id="test_session_1",
        user_id="test_user_1",
    )


@pytest.fixture
def sample_enriched_query():
    return EnrichedQuery(
        original_text="what is M to cm?",
        rewritten_text="Explain conversion of metres to centimetres for CBSE Class 3 student",
        keywords=["metre", "centimetre", "conversion", "Class 3"],
        subject="Mathematics",
        topic="Measurement",
        sub_topic="Metres to Centimetres",
        query_type="conceptual_and_procedural",
    )


@pytest.fixture
def sample_context():
    return ContextObject(
        learner_grade="Class 3",
        learning_styles=["learnstyle:example_driven", "learnstyle:step_by_step", "learnstyle:immediate_feedback"],
        technically_strong_areas=["division", "multiplication"],
        technically_weak_areas=["unit conversion"],
        softskills_strong_areas=["softskill:pattern_mapping"],
        softskills_weak_areas=["softskill:attention_control"],
    )


@pytest.fixture
def sample_retrieval_results():
    return [
        RetrievalResult(
            chunk_id="chunk_1",
            text="1 metre is equal to 100 centimetres. Metre is the standard unit of length.",
            source="NCERT Class 3 Math Ch.6 pg.72",
            chapter="Measurement",
            section="Length",
            score=0.92,
        ),
        RetrievalResult(
            chunk_id="chunk_2",
            text="To convert metres to centimetres, multiply the number of metres by 100.",
            source="NCERT Class 3 Math Ch.6 pg.73",
            chapter="Measurement",
            section="Length",
            score=0.88,
        ),
    ]


@pytest.fixture
def sample_reranked_chunks():
    return [
        RerankedChunk(
            chunk_id="chunk_1",
            text="1 metre is equal to 100 centimetres. Metre is the standard unit of length.",
            source="NCERT Class 3 Math Ch.6 pg.72",
            original_score=0.92,
            rerank_score=0.95,
        ),
        RerankedChunk(
            chunk_id="chunk_2",
            text="To convert metres to centimetres, multiply the number of metres by 100.",
            source="NCERT Class 3 Math Ch.6 pg.73",
            original_score=0.88,
            rerank_score=0.90,
        ),
    ]


@pytest.fixture
def sample_prompt_version():
    return PromptVersion(
        version_id="v3_grade3_standard",
        template=(
            "You are a warm, patient teacher for CBSE Class 3 students. "
            "Explain concepts step-by-step using real-world examples. "
            "Use simple language. Always include a practice question at the end."
        ),
        grade="Class 3",
        variant="standard",
    )
