"""FastAPI application entry point."""

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from api.routes import chat, ingest, keywords, prompts, quiz
from config.architectures import get_architecture
from config.settings import settings

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — startup and shutdown events."""
    arch = get_architecture()
    logger.info(
        "app.startup",
        architecture=arch.name,
        generation_model=settings.generation_model,
        context_model=settings.context_model,
    )
    yield
    logger.info("app.shutdown")


app = FastAPI(
    title="CAPG-RAG",
    description="Context-Aware Prompt Governance System for RAG-based AI Applications",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(chat.router, tags=["Chat"])
app.include_router(quiz.router, tags=["Quiz"])
app.include_router(ingest.router, tags=["Ingestion"])
app.include_router(keywords.router, tags=["Keywords"])
app.include_router(prompts.router, tags=["Prompts"])


@app.get("/health")
async def health():
    arch = get_architecture()
    return {
        "status": "ok",
        "architecture": arch.name,
        "generation_model": settings.generation_model,
    }
