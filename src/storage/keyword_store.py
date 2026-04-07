"""Keyword store — grade+subject → curated NCERT keyword pool.

Keys are "{grade}_{subject}" e.g. "5_Maths", "10_Science".
Values are lists of domain-specific terms extracted from the textbook.

These keywords are injected into the query transform step to improve
retrieval precision beyond what the LLM would guess from the raw query alone.
"""

import structlog

logger = structlog.get_logger()

# In-memory store; production uses Redis or PostgreSQL.
_store: dict[str, list[str]] = {}


def _key(grade: str, subject: str) -> str:
    return f"{grade}_{subject}"


class KeywordStore:

    async def save(self, grade: str, subject: str, keywords: list[str]) -> None:
        """Add or overwrite the keyword pool for a grade+subject combination."""
        k = _key(grade, subject)
        _store[k] = keywords
        logger.info("keyword_store.saved", key=k, count=len(keywords))

    async def lookup(self, grade: str, subject: str) -> list[str]:
        """Return the keyword pool for the given grade+subject. Empty list if not found."""
        k = _key(grade, subject)
        keywords = _store.get(k, [])
        logger.debug("keyword_store.lookup", key=k, found=len(keywords))
        return keywords

    async def list_keys(self) -> list[str]:
        """Return all registered grade+subject keys."""
        return list(_store.keys())
