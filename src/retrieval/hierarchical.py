"""Hierarchical retrieval for Architecture B.

Progressively narrows the search space using the NCERT book structure:
  Level 0: grade + subject     (always applied — scopes to correct knowledge partition)
  Level 1: unit/part           (e.g. "Unit II", "Part I")
  Level 2: chapter             (e.g. chapter_title = "Measurement")
  Level 3: section             (e.g. section_number = "5.1")
  Level 4: subsection          (e.g. subsection_number = "5.1.1")

Each level narrows the candidate set before passing to cross-encoder reranking,
keeping the reranker input small and latency low.
"""

import structlog

from src.models.retrieval import RetrievalResult
from src.retrieval.vector_store import VectorStore

logger = structlog.get_logger()


class HierarchicalRetriever:
    """NCERT-aware hierarchical retrieval (Arch B only).

    Always scopes to grade + subject. Then progressively narrows using
    whatever hierarchy hints the Query Transformation Agent extracted.
    """

    def __init__(self, vector_store: VectorStore):
        self._store = vector_store

    async def search(
        self,
        query: str,
        keywords: list[str],
        grade: str = "",
        subject: str = "",
        unit: str = "",
        chapter_title: str = "",
        section_number: str = "",
        subsection_number: str = "",
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """Hierarchical narrowing search.

        Args:
            query: The enriched query text.
            keywords: Keyword list from the Query Transformation Agent.
            grade: Learner grade — always passed from the learner profile.
            subject: Subject — always passed from the enriched query.
            unit: Unit/Part hint if identifiable from the query (e.g. "Unit II").
            chapter_title: Chapter hint if identifiable (e.g. "Measurement").
            section_number: Section hint if identifiable (e.g. "5.1").
            subsection_number: Subsection hint if identifiable (e.g. "5.1.1").
            top_k: Number of final candidates to return.
        """
        augmented_query = f"{query} {' '.join(keywords)}"

        # Determine deepest available level and log accordingly
        if subsection_number:
            level, label = 4, f"{subject} › {chapter_title} › {section_number} › {subsection_number}"
        elif section_number:
            level, label = 3, f"{subject} › {chapter_title} › {section_number}"
        elif chapter_title:
            level, label = 2, f"{subject} › {chapter_title}"
        elif unit:
            level, label = 1, f"{subject} › {unit}"
        else:
            level, label = 0, f"Grade {grade} › {subject}"

        results = await self._store.search_with_filter(
            query=augmented_query,
            grade=grade,
            subject=subject,
            unit=unit,
            chapter_title=chapter_title,
            section_number=section_number,
            subsection_number=subsection_number,
            top_k=top_k,
        )

        logger.debug(
            "hierarchical.search",
            level=level,
            scope=label,
            results=len(results),
        )
        return results
