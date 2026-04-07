"""Keyword management endpoints — manage the grade+subject keyword store."""

from fastapi import APIRouter
from pydantic import BaseModel

from src.tools.keyword_lookup import get_keyword_store

router = APIRouter()


class KeywordUpsertRequest(BaseModel):
    grade: str          # e.g. "5", "10"
    subject: str        # e.g. "Maths", "Science", "Social Science"
    keywords: list[str]


class KeywordLookupResponse(BaseModel):
    grade: str
    subject: str
    keywords: list[str]
    count: int


@router.post("/keywords", status_code=201)
async def upsert_keywords(request: KeywordUpsertRequest):
    """Add or overwrite the keyword pool for a grade+subject combination.

    Keywords are extracted from NCERT textbooks and stored as a flat list.
    They are injected into the Context/Query Transform agent prompt at query time.
    """
    store = get_keyword_store()
    await store.save(request.grade, request.subject, request.keywords)
    return {
        "status": "saved",
        "key": f"{request.grade}_{request.subject}",
        "count": len(request.keywords),
    }


@router.get("/keywords/{grade}/{subject}", response_model=KeywordLookupResponse)
async def get_keywords(grade: str, subject: str):
    """Look up the keyword pool for a specific grade+subject combination."""
    store = get_keyword_store()
    keywords = await store.lookup(grade, subject)
    return KeywordLookupResponse(
        grade=grade,
        subject=subject,
        keywords=keywords,
        count=len(keywords),
    )


@router.get("/keywords")
async def list_keyword_keys():
    """List all registered grade+subject combinations."""
    store = get_keyword_store()
    keys = await store.list_keys()
    return {"keys": keys}
