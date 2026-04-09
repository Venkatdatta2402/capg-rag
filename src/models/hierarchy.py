"""Hierarchy node models for level-by-level traversal.

One node per unique (level, scope) combination in the knowledge base.
Stored in a separate Qdrant collection (capg_hierarchy).
The node's summary is embedded for vector similarity at traversal time.

node_type values: "unit" | "chapter" | "section" | "subsection"
"""

from pydantic import BaseModel, Field


class HierarchyNode(BaseModel):
    node_id: str            # e.g. "Grade5_Maths_unit_Unit-II"
    node_type: str          # unit | chapter | section | subsection
    title: str              # human-readable label used for title match boost
    summary: str            # LLM-generated; embedded for vector similarity

    # Aggregated from constituent chunks — used for keyword/concept overlap scoring
    keywords: list[str] = Field(default_factory=list)
    concepts: list[str] = Field(default_factory=list)

    # Scope — filled down to the level this node lives at
    grade: str = ""
    subject: str = ""
    unit: str = ""
    chapter_title: str = ""
    section_number: str = ""
    section_title: str = ""
    subsection_number: str = ""
    subsection_title: str = ""

    score: float = 0.0      # filled at retrieval time
