"""Tests for pipeline agents."""

import pytest
from unittest.mock import AsyncMock

from src.agents.query_transform import QueryTransformAgent
from src.agents.judge import JudgeAgent


class TestQueryTransformAgent:
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
            "KEYWORDS: metre, centimetre, conversion, length\n"
            "REWRITTEN_QUERY: Explain conversion of metres to centimetres for CBSE Class 5"
        )
        return llm

    @pytest.mark.asyncio
    async def test_transform_produces_keywords(self, mock_llm, sample_query_input):
        agent = QueryTransformAgent(mock_llm)
        enriched = await agent.run(sample_query_input, grade="Class 5")

        assert "metre" in enriched.keywords
        assert enriched.subject == "Mathematics"
        assert enriched.rewritten_text != sample_query_input.query_text


class TestJudgeAgent:
    @pytest.fixture
    def mock_llm(self):
        llm = AsyncMock()
        llm.model = "gpt-4o-mini"
        return llm

    @pytest.mark.asyncio
    async def test_evaluate_understood(self, mock_llm):
        mock_llm.generate.return_value = (
            "COT_STEP_1: Core concept: 1m = 100cm\n"
            "COT_STEP_2: Learner said 200 cm\n"
            "COT_STEP_3: Contains key concept\n"
            "COT_STEP_4: Genuine understanding\n"
            "VERDICT: UNDERSTOOD\n"
            "RATIONALE: Correct\n"
            "RETRIEVAL_FEEDBACK: none"
        )
        judge = JudgeAgent(mock_llm)
        verdict = await judge.evaluate(
            question="How many cm is 2 metres?",
            learner_response="200 cm",
            explanation="1 metre = 100 centimetres",
            topic="unit conversion",
        )
        assert verdict.verdict == "UNDERSTOOD"
        assert verdict.retrieval_feedback == ""

    @pytest.mark.asyncio
    async def test_evaluate_not_understood_has_feedback(self, mock_llm):
        mock_llm.generate.return_value = (
            "COT_STEP_1: Core concept: osmosis mechanism\n"
            "COT_STEP_2: Learner said water moves\n"
            "COT_STEP_3: Missing semipermeable membrane detail\n"
            "COT_STEP_4: Surface guess\n"
            "VERDICT: NOT_UNDERSTOOD\n"
            "RATIONALE: Missing key mechanism\n"
            "RETRIEVAL_FEEDBACK: role of semipermeable membrane in osmosis"
        )
        judge = JudgeAgent(mock_llm)
        verdict = await judge.evaluate(
            question="What drives osmosis?",
            learner_response="Water moves",
            explanation="Osmosis occurs through a semipermeable membrane.",
            topic="osmosis",
        )
        assert verdict.verdict == "NOT_UNDERSTOOD"
        assert "semipermeable" in verdict.retrieval_feedback
