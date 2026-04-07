"""Tests for Architecture A agents."""

import pytest
from unittest.mock import AsyncMock

from src.agents.arch_a.context_rephrase import ContextRephraseAgent
from src.agents.arch_a.judge import SeparateJudgeAgent
from src.models.query import QueryInput
from src.storage.user_profile import UserProfileStore
from src.storage.session_memory import SessionMemoryStore


class TestContextRephraseAgent:
    """Tests for the combined Context & Rephrase Agent."""

    @pytest.fixture
    def mock_llm(self):
        llm = AsyncMock()
        llm.model = "gpt-4o-mini"
        llm.generate.return_value = (
            "SUBJECT: Mathematics\n"
            "TOPIC: Measurement\n"
            "SUB_TOPIC: Metres to Centimetres\n"
            "QUERY_TYPE: conceptual_and_procedural\n"
            "REWRITTEN_QUERY: Explain conversion of metres to centimetres for CBSE Class 3\n"
            "KEYWORDS: metre, centimetre, conversion"
        )
        return llm

    @pytest.mark.asyncio
    async def test_produces_enriched_query_and_context(
        self, mock_llm, sample_query_input, sample_profile
    ):
        profile_store = UserProfileStore()
        session_store = SessionMemoryStore()
        await profile_store.save(sample_profile)

        agent = ContextRephraseAgent(mock_llm, profile_store, session_store)
        enriched, context = await agent.run(sample_query_input)

        assert enriched.subject == "Mathematics"
        assert enriched.topic == "Measurement"
        assert len(enriched.keywords) > 0
        assert context.learner_grade == "Class 3"
        assert context.comprehension_level == "low-medium"


class TestSeparateJudgeAgent:
    """Tests for the separate judge agent."""

    @pytest.fixture
    def mock_llm(self):
        llm = AsyncMock()
        llm.model = "gpt-4o-mini"
        return llm

    @pytest.mark.asyncio
    async def test_evaluate_understood(self, mock_llm):
        mock_llm.generate.return_value = (
            "COT_STEP_1: Learner must demonstrate 1m = 100cm\n"
            "COT_STEP_2: Learner said 200 cm\n"
            "COT_STEP_3: Correct application of multiplication\n"
            "COT_STEP_4: Not a surface guess\n"
            "VERDICT: UNDERSTOOD\n"
            "RATIONALE: Correct application of conversion"
        )

        judge = SeparateJudgeAgent(mock_llm)
        verdict = await judge.evaluate(
            question="How many cm is 2 metres?",
            learner_response="200 cm",
            explanation="1 metre = 100 centimetres",
            topic="unit conversion",
        )
        assert verdict.verdict == "UNDERSTOOD"

    @pytest.mark.asyncio
    async def test_evaluate_not_understood(self, mock_llm):
        mock_llm.generate.return_value = (
            "COT_STEP_1: Learner must demonstrate 1m = 100cm\n"
            "COT_STEP_2: Learner said 100\n"
            "COT_STEP_3: Just repeated the definition\n"
            "COT_STEP_4: Surface guess\n"
            "VERDICT: NOT_UNDERSTOOD\n"
            "RATIONALE: Learner repeated definition without applying it"
        )

        judge = SeparateJudgeAgent(mock_llm)
        verdict = await judge.evaluate(
            question="How many cm is 2 metres?",
            learner_response="100",
            explanation="1 metre = 100 centimetres",
            topic="unit conversion",
        )
        assert verdict.verdict == "NOT_UNDERSTOOD"
