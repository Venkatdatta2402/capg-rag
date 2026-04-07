"""Feedback record storage — the bridge between Loop A (runtime) and Loop B (governance)."""

import structlog

from src.models.feedback import FeedbackRecord

logger = structlog.get_logger()

# In-memory store; production uses PostgreSQL.
_records: list[FeedbackRecord] = []


class FeedbackStore:
    """Stores structured feedback records from every interaction."""

    async def write(self, record: FeedbackRecord) -> None:
        """Write a feedback record after a judge verdict."""
        _records.append(record)
        logger.info(
            "feedback.written",
            prompt_version=record.prompt_version,
            verdict=record.verdict,
            architecture=record.architecture,
        )

    async def get_records(
        self,
        prompt_version: str | None = None,
        architecture: str | None = None,
        limit: int = 500,
    ) -> list[FeedbackRecord]:
        """Query feedback records with optional filters."""
        results = _records
        if prompt_version:
            results = [r for r in results if r.prompt_version == prompt_version]
        if architecture:
            results = [r for r in results if r.architecture == architecture]
        return results[-limit:]

    async def count(self) -> int:
        """Return total number of feedback records."""
        return len(_records)

    async def get_records_since_last_governance_run(self) -> list[FeedbackRecord]:
        """Return records accumulated since the last governance pipeline run.

        In production, this would track a watermark or timestamp.
        For now, returns all records.
        """
        return list(_records)
