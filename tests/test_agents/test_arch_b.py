"""Tests for Architecture B agents."""

import pytest
from unittest.mock import AsyncMock

from src.agents.arch_b.query_transform import QueryTransformAgent
from src.agents.arch_b.context_builder import ContextObjectBuilder
from src.agents.arch_b.judge import UnifiedJudgeAgent
from src.models.query import QueryInput


class TestQueryTransformAgent:
    """Tests for the Query Transformation Agent."""

    @pytest.fixture
    def mock_llm(self):
        llm = AsyncMock()
        llm.model = "gpt-4o-mini"
        llm.generate.return_value = (
            "SUBJECT: Mathematics\n"
            "TOPIC: Measurement\n"
            "SUB_TOPIC: Metres to Centimetres\n"
            "CHAPTER_HINT: Measurement\n"
            "SECTION_HINT: Length\n"
            "QUERY_TYPE: conceptual_and_procedural\n"
            "KEYWORDS: metre, centimetre, conversion, length, Class 3\n"
            "REWRITTEN_QUERY: Explain conversion of metres to centimetres for CBSE Class 3"
        )
        return llm

    @pytest.mark.asyncio
    async def test_transform_produces_keywords(self, mock_llm, sample_query_input, sample_profile):
        agent = QueryTransformAgent(mock_llm)
        enriched = await agent.run(sample_query_input, sample_profile)

        assert "metre" in enriched.keywords
        assert enriched.subject == "Mathematics"
        assert enriched.rewritten_text != sample_query_input.query_text


class TestContextObjectBuilder:
    """Tests for the deterministic Context Object Builder."""

    def test_builds_context_from_profile(self, sample_profile, sample_session, sample_enriched_query):
        builder = ContextObjectBuilder()
        context = builder.build(sample_profile, sample_session, sample_enriched_query)

        assert context.learner_grade == "Class 3"
        assert context.comprehension_level == "low-medium"
        assert context.learning_style == "example-driven"
        assert "unit conversion" in context.weak_areas

    def test_comprehension_levels(self, sample_profile, sample_session, sample_enriched_query):
        builder = ContextObjectBuilder()

        sample_profile.comprehension_score = 2.0
        ctx = builder.build(sample_profile, sample_session, sample_enriched_query)
        assert ctx.comprehension_level == "low"

        sample_profile.comprehension_score = 9.0
        ctx = builder.build(sample_profile, sample_session, sample_enriched_query)
        assert ctx.comprehension_level == "high"


class TestUnifiedJudgeAgent:
    """Tests for the unified judge (reuses generation model)."""

    @pytest.fixture
    def mock_llm(self):
        llm = AsyncMock()
        llm.model = "gpt-4.1"
        return llm

    @pytest.mark.asyncio
    async def test_evaluate(self, mock_llm):
        mock_llm.generate.return_value = (
            "COT_STEP_1: Core concept: 1m = 100cm and multiplication\n"
            "COT_STEP_2: Learner said 200 cm\n"
            "COT_STEP_3: Contains key concept\n"
            "COT_STEP_4: Genuine understanding\n"
            "VERDICT: UNDERSTOOD\n"
            "RATIONALE: Correct"
        )

        judge = UnifiedJudgeAgent(mock_llm)
        verdict = await judge.evaluate(
            question="How many cm is 2 metres?",
            learner_response="200 cm",
            explanation="1 metre = 100 centimetres",
            topic="unit conversion",
        )
        assert verdict.verdict == "UNDERSTOOD"
