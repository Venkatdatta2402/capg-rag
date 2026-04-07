"""Qdrant vector store client for dense retrieval."""

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from config.settings import settings
from src.models.retrieval import RetrievalResult

logger = structlog.get_logger()


class VectorStore:
    """Dense retrieval using Qdrant."""

    def __init__(self):
        self._client = AsyncQdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        self._encoder = SentenceTransformer(settings.embedding_model)
        self._collection = settings.qdrant_collection

    async def ensure_collection(self, vector_size: int = 384) -> None:
        """Create the collection if it doesn't exist."""
        collections = await self._client.get_collections()
        names = [c.name for c in collections.collections]
        if self._collection not in names:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            logger.info("vector_store.collection_created", name=self._collection)

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode texts into dense vectors."""
        return self._encoder.encode(texts, convert_to_numpy=True).tolist()

    async def upsert(self, chunk_id: str, text: str, metadata: dict) -> None:
        """Index a single chunk."""
        vector = self.encode([text])[0]
        point = PointStruct(
            id=hash(chunk_id) % (2**63),
            vector=vector,
            payload={"chunk_id": chunk_id, "text": text, **metadata},
        )
        await self._client.upsert(collection_name=self._collection, points=[point])

    async def search(
        self,
        query: str,
        grade: str = "",
        subject: str = "",
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Semantic search against the vector store.

        grade and subject are always applied as filters when provided —
        they scope every search to the correct knowledge base partition.
        """
        return await self.search_with_filter(
            query=query, grade=grade, subject=subject, top_k=top_k
        )

    async def search_with_filter(
        self,
        query: str,
        grade: str = "",
        subject: str = "",
        unit: str = "",
        chapter_title: str = "",
        section_number: str = "",
        subsection_number: str = "",
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Filtered semantic search — supports the full NCERT hierarchy.

        Filter precedence (all non-empty values are ANDed):
          grade + subject  →  unit  →  chapter_title  →  section_number  →  subsection_number
        """
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        top_k = top_k or settings.top_k_retrieval
        vector = self.encode([query])[0]

        conditions = []
        # Scope filters — always applied when present
        if grade:
            conditions.append(FieldCondition(key="grade", match=MatchValue(value=grade)))
        if subject:
            conditions.append(FieldCondition(key="subject", match=MatchValue(value=subject)))
        # Hierarchy filters — progressively narrow
        if unit:
            conditions.append(FieldCondition(key="unit", match=MatchValue(value=unit)))
        if chapter_title:
            conditions.append(FieldCondition(key="chapter_title", match=MatchValue(value=chapter_title)))
        if section_number:
            conditions.append(FieldCondition(key="section_number", match=MatchValue(value=section_number)))
        if subsection_number:
            conditions.append(FieldCondition(key="subsection_number", match=MatchValue(value=subsection_number)))

        query_filter = Filter(must=conditions) if conditions else None

        hits = await self._client.search(
            collection_name=self._collection,
            query_vector=vector,
            query_filter=query_filter,
            limit=top_k,
        )
        return [self._hit_to_result(hit) for hit in hits]

    def _hit_to_result(self, hit) -> RetrievalResult:
        """Convert a Qdrant search hit to a RetrievalResult."""
        p = hit.payload or {}
        return RetrievalResult(
            chunk_id=p.get("chunk_id", ""),
            text=p.get("text", ""),
            source=p.get("source", ""),
            grade=p.get("grade", ""),
            subject=p.get("subject", ""),
            unit=p.get("unit", ""),
            chapter_number=p.get("chapter_number", ""),
            chapter_title=p.get("chapter_title", ""),
            section_number=p.get("section_number", ""),
            section_title=p.get("section_title", ""),
            subsection_number=p.get("subsection_number", ""),
            subsection_title=p.get("subsection_title", ""),
            score=hit.score,
            metadata=p,
        )
