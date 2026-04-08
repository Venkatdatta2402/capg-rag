"""FastAPI application entry point."""

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from api.routes import chat, ingest, keywords, prompts, quiz, session
from config.settings import settings

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "app.startup",
        generation_model=settings.generation_model,
        context_model=settings.context_model,
        judge_model=settings.judge_model,
    )
    yield
    logger.info("app.shutdown")


app = FastAPI(
    title="PG-CARAG",
    description="Context-Aware Prompt Governance RAG System for CBSE/NCERT",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(chat.router, tags=["Chat"])
app.include_router(quiz.router, tags=["Quiz"])
app.include_router(session.router, tags=["Session"])
app.include_router(ingest.router, tags=["Ingestion"])
app.include_router(keywords.router, tags=["Keywords"])
app.include_router(prompts.router, tags=["Prompts"])


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "generation_model": settings.generation_model,
        "context_model": settings.context_model,
        "judge_model": settings.judge_model,
    }
