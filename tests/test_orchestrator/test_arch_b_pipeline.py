"""Tests for Architecture B pipeline components."""

import pytest

from src.orchestrator.factory import get_orchestrator
from src.orchestrator.retry import RetryManager
from src.models.feedback import JudgeVerdict
from src.storage.session_memory import SessionMemoryStore
from src.storage.user_profile import UserProfileStore


class TestRetryManager:
    """Tests for the adaptive retry loop."""

    @pytest.mark.asyncio
    async def test_should_retry_when_not_understood(self):
        manager = RetryManager(SessionMemoryStore(), UserProfileStore(), max_retries=3)
        verdict = JudgeVerdict(verdict="NOT_UNDERSTOOD")
        assert await manager.should_retry("s1", "u1", verdict) is True

    @pytest.mark.asyncio
    async def test_should_not_retry_when_understood(self):
        manager = RetryManager(SessionMemoryStore(), UserProfileStore(), max_retries=3)
        verdict = JudgeVerdict(verdict="UNDERSTOOD")
        assert await manager.should_retry("s1", "u1", verdict) is False

    @pytest.mark.asyncio
    async def test_prepare_retry_increments_count(self):
        session_store = SessionMemoryStore()
        manager = RetryManager(session_store, UserProfileStore(), max_retries=3)

        count = await manager.prepare_retry("s1", "u1", "unit conversion")
        assert count == 1

        count = await manager.prepare_retry("s1", "u1", "unit conversion")
        assert count == 2
