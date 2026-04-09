"""Qdrant store for HierarchyNodes — one node per unique level scope.

Summaries are embedded for vector similarity traversal at each hierarchy level.

Scoring formula:
  score = w_v * vector_similarity
        + w_kw * keyword_overlap
        + w_ct * concept_overlap
        + w_tm * title_match

Base weights: w_v=0.90, w_kw=0.05, w_ct=0.03, w_tm=0.02

keyword_overlap  = |query_keywords ∩ node_keywords| / len(query_keywords)
concept_overlap  = |query_concepts ∩ node_concepts| / len(query_concepts)
title_match      = 1.0 if any query token appears in node title, else 0.0

If query_keywords or query_concepts is empty the corresponding weight is set
to 0 and all weights are renormalised so they sum to 1.0.
"""

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from config.settings import settings
from src.models.hierarchy import HierarchyNode

logger = structlog.get_logger()

_BASE_W_VECTOR  = 0.90
_BASE_W_KEYWORD = 0.05
_BASE_W_CONCEPT = 0.03
_BASE_W_TITLE   = 0.02


def _weights(has_keywords: bool, has_concepts: bool) -> tuple[float, float, float, float]:
    """Return (w_vector, w_keyword, w_concept, w_title) renormalised to sum to 1."""
    w_v  = _BASE_W_VECTOR
    w_kw = _BASE_W_KEYWORD if has_keywords else 0.0
    w_ct = _BASE_W_CONCEPT if has_concepts else 0.0
    w_tm = _BASE_W_TITLE
    total = w_v + w_kw + w_ct + w_tm
    return w_v / total, w_kw / total, w_ct / total, w_tm / total


def _overlap(query_set: set[str], node_set: set[str]) -> float:
    """Precision-style overlap: intersection / len(query_set)."""
    if not query_set:
        return 0.0
    return len(query_set & node_set) / len(query_set)


def _title_match(query_tokens: set[str], title: str) -> float:
    return 1.0 if any(t in title.lower() for t in query_tokens) else 0.0


def _score(
    vector_sim: float,
    q_keywords: set[str],
    q_concepts: set[str],
    q_tokens: set[str],
    node: HierarchyNode,
) -> float:
    w_v, w_kw, w_ct, w_tm = _weights(bool(q_keywords), bool(q_concepts))
    node_keywords = {kw.lower() for kw in node.keywords}
    node_concepts = {ct.lower() for ct in node.concepts}
    return (
        w_v  * vector_sim
        + w_kw * _overlap(q_keywords, node_keywords)
        + w_ct * _overlap(q_concepts, node_concepts)
        + w_tm * _title_match(q_tokens, node.title)
    )


class HierarchyStore:
    """Embed and search HierarchyNodes in the capg_hierarchy Qdrant collection."""

    def __init__(self):
        self._client = AsyncQdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        self._encoder = SentenceTransformer(settings.embedding_model)
        self._collection = settings.qdrant_hierarchy_collection

    async def ensure_collection(self, vector_size: int = 384) -> None:
        collections = await self._client.get_collections()
        names = [c.name for c in collections.collections]
        if self._collection not in names:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            logger.info("hierarchy_store.collection_created", name=self._collection)

    def _encode(self, texts: list[str]) -> list[list[float]]:
        return self._encoder.encode(texts, convert_to_numpy=True).tolist()

    async def upsert_batch(self, nodes: list[HierarchyNode]) -> None:
        """Embed all summaries in one batch and upsert."""
        if not nodes:
            return
        vectors = self._encode([n.summary for n in nodes])
        points = [
            PointStruct(
                id=hash(n.node_id) % (2**63),
                vector=vectors[i],
                payload={
                    "node_id": n.node_id,
                    "node_type": n.node_type,
                    "title": n.title,
                    "summary": n.summary,
                    "grade": n.grade,
                    "subject": n.subject,
                    "unit": n.unit,
                    "chapter_title": n.chapter_title,
                    "section_number": n.section_number,
                    "section_title": n.section_title,
                    "subsection_number": n.subsection_number,
                    "subsection_title": n.subsection_title,
                    "keywords": n.keywords,
                    "concepts": n.concepts,
                },
            )
            for i, n in enumerate(nodes)
        ]
        await self._client.upsert(collection_name=self._collection, points=points)
        logger.info("hierarchy_store.upserted", count=len(nodes))

    async def search(
        self,
        query: str,
        query_keywords: list[str],
        query_concepts: list[str],
        node_type: str,
        grade: str,
        subject: str,
        units: list[str] | None = None,
        chapter_titles: list[str] | None = None,
        section_numbers: list[str] | None = None,
        top_k: int = 3,
    ) -> list[HierarchyNode]:
        """Search for top_k nodes at a given level, scored by the composite formula.

        Parent-level filters accept lists (MatchAny) so all top-3 candidates
        from the previous level are included in the search space.

        Fetches top_k * 2 candidates by vector similarity, re-scores using the
        full formula (with renormalisation when keyword/concept sets are empty),
        and returns top_k nodes ordered by combined score.
        """
        from qdrant_client.models import MatchAny

        vector = self._encode([query])[0]

        conditions = [
            FieldCondition(key="node_type", match=MatchValue(value=node_type)),
            FieldCondition(key="grade",     match=MatchValue(value=grade)),
            FieldCondition(key="subject",   match=MatchValue(value=subject)),
        ]
        if units:
            conditions.append(FieldCondition(key="unit",           match=MatchAny(any=units)))
        if chapter_titles:
            conditions.append(FieldCondition(key="chapter_title",  match=MatchAny(any=chapter_titles)))
        if section_numbers:
            conditions.append(FieldCondition(key="section_number", match=MatchAny(any=section_numbers)))

        hits = await self._client.search(
            collection_name=self._collection,
            query_vector=vector,
            query_filter=Filter(must=conditions),
            limit=top_k * 2,
        )

        q_keywords = {kw.lower() for kw in query_keywords}
        q_concepts = {ct.lower() for ct in query_concepts}
        q_tokens   = set(query.lower().split())

        nodes: list[HierarchyNode] = []
        for hit in hits:
            p = hit.payload or {}
            node = HierarchyNode(
                node_id=p.get("node_id", ""),
                node_type=p.get("node_type", ""),
                title=p.get("title", ""),
                summary=p.get("summary", ""),
                grade=p.get("grade", ""),
                subject=p.get("subject", ""),
                unit=p.get("unit", ""),
                chapter_title=p.get("chapter_title", ""),
                section_number=p.get("section_number", ""),
                section_title=p.get("section_title", ""),
                subsection_number=p.get("subsection_number", ""),
                subsection_title=p.get("subsection_title", ""),
                keywords=p.get("keywords", []),
                concepts=p.get("concepts", []),
            )
            node.score = _score(hit.score, q_keywords, q_concepts, q_tokens, node)
            nodes.append(node)

        nodes.sort(key=lambda n: n.score, reverse=True)
        logger.debug(
            "hierarchy_store.search",
            node_type=node_type, candidates=len(hits), returned=min(top_k, len(nodes)),
        )
        return nodes[:top_k]
