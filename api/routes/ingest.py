"""Ingestion endpoints for loading knowledge base documents."""

from fastapi import APIRouter, Form, UploadFile, File

from pydantic import BaseModel

router = APIRouter()


class IngestResponse(BaseModel):
    status: str
    chunks_indexed: int
    message: str


@router.post("/ingest", response_model=IngestResponse)
async def ingest_pdf(
    file: UploadFile = File(...),
    grade: str = Form(...),           # e.g. "5" or "10"
    subject: str = Form(...),         # e.g. "Maths", "Science", "Social Science"
    unit: str = Form(""),             # e.g. "Unit II", "Part I" — optional
    chapter_number: str = Form(""),   # e.g. "6"
    chapter_title: str = Form(""),    # e.g. "Measurement"
):
    """Ingest a PDF document into the knowledge base.

    The caller must provide grade and subject — these scope every chunk
    to the correct knowledge partition for filtering at query time.
    Unit, chapter_number, and chapter_title are optional but improve
    hierarchical retrieval precision.

    Section and subsection metadata are auto-extracted from the PDF
    by detecting decimal-numbered headings (e.g. "5.1", "5.1.1").
    """
    from src.ingestion.pdf_parser import PDFParser
    from src.ingestion.chunker import Chunker
    from src.ingestion.indexer import Indexer

    content = await file.read()
    filename = file.filename or "unknown"
    source = f"NCERT_Grade{grade}_{subject}_{filename}"

    parser = PDFParser()
    pages = parser.parse(content)

    chunker = Chunker()
    chunks = chunker.chunk(
        pages,
        source=source,
        grade=grade,
        subject=subject,
        unit=unit,
        chapter_number=chapter_number,
        chapter_title=chapter_title,
    )

    indexer = Indexer()
    count = await indexer.index(chunks)

    return IngestResponse(
        status="success",
        chunks_indexed=count,
        message=f"Indexed {count} chunks from {filename} (Grade {grade} {subject})",
    )
