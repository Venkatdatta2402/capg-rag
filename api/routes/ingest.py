"""Ingestion endpoint — parse, enrich, embed and index NCERT PDF chunks."""

from fastapi import APIRouter, Form, UploadFile, File
from pydantic import BaseModel

from config.settings import settings
from src.ingestion.chunk_enricher import ChunkEnricher
from src.ingestion.chunker import Chunker
from src.ingestion.hierarchy_builder import HierarchyBuilder
from src.ingestion.indexer import Indexer
from src.ingestion.pdf_parser import PDFParser
from src.llm.factory import get_llm_client
from src.retrieval.hierarchy_store import HierarchyStore

router = APIRouter()


class IngestResponse(BaseModel):
    status: str
    chunks_indexed: int
    hierarchy_nodes_indexed: int
    message: str


@router.post("/ingest", response_model=IngestResponse)
async def ingest_pdf(
    file: UploadFile = File(...),
    grade: str = Form(...),           # e.g. "5" or "10"
    subject: str = Form(...),         # e.g. "Maths", "Science"
    unit: str = Form(""),             # e.g. "Unit II" — optional
    chapter_number: str = Form(""),   # e.g. "6"
    chapter_title: str = Form(""),    # e.g. "Measurement"
):
    """Ingest a PDF into the knowledge base.

    1. Parse PDF → pages
    2. Chunk pages (structure-aware, 450-word limit)
    3. Enrich chunks with LLM-generated keywords (5-8) and concepts (2-4)
    4. Index enriched chunks in Qdrant capg_knowledge
    5. Build hierarchy nodes (one per unique level scope) with LLM summaries
    6. Index hierarchy nodes in Qdrant capg_hierarchy
    """
    content = await file.read()
    filename = file.filename or "unknown"
    source = f"NCERT_Grade{grade}_{subject}_{filename}"

    pages = PDFParser().parse(content)

    chunks = Chunker().chunk(
        pages,
        source=source,
        grade=grade,
        subject=subject,
        unit=unit,
        chapter_number=chapter_number,
        chapter_title=chapter_title,
    )

    llm = get_llm_client(settings.context_provider, settings.context_model)

    enriched_chunks = await ChunkEnricher(llm).enrich_all(chunks)
    chunk_count = await Indexer().index(enriched_chunks)

    nodes = await HierarchyBuilder(llm).build(enriched_chunks)
    hierarchy_store = HierarchyStore()
    await hierarchy_store.ensure_collection()
    await hierarchy_store.upsert_batch(nodes)

    return IngestResponse(
        status="success",
        chunks_indexed=chunk_count,
        hierarchy_nodes_indexed=len(nodes),
        message=(
            f"Indexed {chunk_count} chunks and {len(nodes)} hierarchy nodes "
            f"from {filename} (Grade {grade} {subject})"
        ),
    )
