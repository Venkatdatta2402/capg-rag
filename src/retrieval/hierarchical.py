"""Hard hierarchical retrieval with summary-based scope traversal.

Traversal strategy (top-3 at every level, never exponential):
  Level 1 — unit:       no parent filter → top-3 units
  Level 2 — chapter:    unit IN [u1,u2,u3] → top-3 chapters across all 3 units
  Level 3 — section:    chapter_title IN [c1,c2,c3] → top-3 sections across all 3 chapters
  Level 4 — subsection: section_number IN [s1,s2,s3] → top-3 subsections across all 3 sections

Final chunk retrieval: parallel dense search for each of the top-3 deepest-level
scopes, merged with sparse (BM25) via Reciprocal Rank Fusion, deduplicated.

Any level absent from the hierarchy collection is skipped.
"""

import asyncio

import structlog

from src.models.retrieval import RetrievalResult
from src.retrieval.hierarchy_store import HierarchyStore
from src.retrieval.sparse import SparseRetriever
from src.retrieval.vector_store import VectorStore

logger = structlog.get_logger()

_LEVEL_TOP_K = 3
_RRF_K = 60


class HierarchicalRetriever:
    """Step-by-step hierarchical retrieval with RRF fusion at the final level."""

    def __init__(self, vector_store: VectorStore, sparse_retriever: SparseRetriever):
        self._store = vector_store
        self._sparse = sparse_retriever
        self._hierarchy = HierarchyStore()

    async def search(
        self,
        query: str,
        keywords: list[str],
        grade: str,
        subject: str,
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        augmented = f"{query} {' '.join(keywords)}" if keywords else query
        q_keywords = keywords
        q_concepts: list[str] = []

        def _hs(node_type: str, **kw):
            return self._hierarchy.search(
                query=augmented,
                query_keywords=q_keywords,
                query_concepts=q_concepts,
                node_type=node_type,
                grade=grade,
                subject=subject,
                top_k=_LEVEL_TOP_K,
                **kw,
            )

        # Level 1: unit — no parent filter
        unit_nodes = await _hs("unit")

        # Level 2: chapter — filter unit IN top-3 units
        chapter_nodes = []
        if unit_nodes:
            chapter_nodes = await _hs("chapter", units=[n.unit for n in unit_nodes])

        # Level 3: section — filter chapter_title IN top-3 chapters
        section_nodes = []
        if chapter_nodes:
            section_nodes = await _hs(
                "section",
                chapter_titles=[n.chapter_title for n in chapter_nodes],
            )

        # Level 4: subsection — filter section_number IN top-3 sections
        subsection_nodes = []
        if section_nodes:
            subsection_nodes = await _hs(
                "subsection",
                section_numbers=[n.section_number for n in section_nodes],
            )

        # Build final scope list from deepest level that returned results.
        # Each node already carries its full scope fields.
        if subsection_nodes:
            final_scopes = [
                {
                    "unit": n.unit,
                    "chapter_title": n.chapter_title,
                    "section_number": n.section_number,
                    "subsection_number": n.subsection_number,
                }
                for n in subsection_nodes
            ]
            deepest = "subsection"
        elif section_nodes:
            final_scopes = [
                {
                    "unit": n.unit,
                    "chapter_title": n.chapter_title,
                    "section_number": n.section_number,
                }
                for n in section_nodes
            ]
            deepest = "section"
        elif chapter_nodes:
            final_scopes = [
                {"unit": n.unit, "chapter_title": n.chapter_title}
                for n in chapter_nodes
            ]
            deepest = "chapter"
        elif unit_nodes:
            final_scopes = [{"unit": n.unit} for n in unit_nodes]
            deepest = "unit"
        else:
            final_scopes = [{}]
            deepest = "grade+subject"

        logger.debug(
            "hierarchical.scope_resolved",
            grade=grade, subject=subject,
            deepest=deepest, scope_count=len(final_scopes),
        )

        return await self._final_retrieve(augmented, grade, subject, final_scopes, top_k)

    # ------------------------------------------------------------------
    # Final retrieval: parallel dense per scope + sparse, fused via RRF
    # ------------------------------------------------------------------

    async def _final_retrieve(
        self,
        query: str,
        grade: str,
        subject: str,
        scopes: list[dict],
        top_k: int,
    ) -> list[RetrievalResult]:
        dense_lists: list[list[RetrievalResult]] = await asyncio.gather(*[
            self._store.search_with_filter(
                query=query,
                grade=grade,
                subject=subject,
                unit=s.get("unit", ""),
                chapter_title=s.get("chapter_title", ""),
                section_number=s.get("section_number", ""),
                subsection_number=s.get("subsection_number", ""),
                top_k=top_k * 2,
            )
            for s in scopes
        ])

        sparse_candidates = self._sparse.search(query, top_k=top_k * 2)
        sparse = [
            r for r in sparse_candidates
            if any(_in_scope(r, grade, subject, s) for s in scopes)
        ]

        rrf: dict[str, float] = {}
        chunk_map: dict[str, RetrievalResult] = {}

        for dense in dense_lists:
            for rank, r in enumerate(dense):
                rrf[r.chunk_id] = rrf.get(r.chunk_id, 0) + 1 / (_RRF_K + rank + 1)
                chunk_map[r.chunk_id] = r

        for rank, r in enumerate(sparse):
            rrf[r.chunk_id] = rrf.get(r.chunk_id, 0) + 1 / (_RRF_K + rank + 1)
            chunk_map.setdefault(r.chunk_id, r)

        sorted_ids = sorted(rrf, key=lambda cid: rrf[cid], reverse=True)[:top_k]
        results = []
        seen: set[str] = set()
        for cid in sorted_ids:
            if cid not in seen:
                seen.add(cid)
                r = chunk_map[cid]
                r.score = rrf[cid]
                results.append(r)

        logger.debug(
            "hierarchical.final_retrieve",
            scopes=len(scopes),
            dense_total=sum(len(d) for d in dense_lists),
            sparse=len(sparse),
            returned=len(results),
        )
        return results


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _in_scope(
    result: RetrievalResult,
    grade: str,
    subject: str,
    scope: dict,
) -> bool:
    meta = result.metadata
    if grade and meta.get("grade", "") != grade:
        return False
    if subject and meta.get("subject", "") != subject:
        return False
    for field in ("unit", "chapter_title", "section_number", "subsection_number"):
        required = scope.get(field, "")
        if required and meta.get(field, "") != required:
            return False
    return True
