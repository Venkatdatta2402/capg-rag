"""LangGraph tool — looks up domain keywords from the keyword store.

Attached to the Context agent (Arch A) and Query Transform agent (Arch B).
Called deterministically from the pipeline node (not via LLM function calling)
so the agent always receives the right keyword pool before it rewrites the query.

Defined as a LangGraph @tool so it can also be plugged into a ReAct agent
loop in future without any changes.
"""

from langchain_core.tools import tool

from src.storage.keyword_store import KeywordStore

# Module-level store instance shared across tool calls
_store = KeywordStore()


@tool
async def lookup_domain_keywords(grade: str, subject: str) -> list[str]:
    """Look up the curated NCERT keyword pool for a grade and subject combination.

    Args:
        grade: Learner grade string, e.g. "5" or "10".
        subject: Subject name, e.g. "Maths", "Science", "Social Science".

    Returns:
        List of domain-specific keywords extracted from the textbook.
        Returns an empty list if no keywords are registered for this combination.
    """
    return await _store.lookup(grade, subject)


async def invoke_keyword_lookup(grade: str, subject: str) -> list[str]:
    """Direct async call for use inside pipeline nodes.

    Wraps the @tool so pipeline nodes don't need to deal with
    LangChain's ToolMessage format — they just get the keyword list.
    """
    return await _store.lookup(grade, subject)


def get_keyword_store() -> KeywordStore:
    """Return the shared keyword store instance (for dependency injection)."""
    return _store
