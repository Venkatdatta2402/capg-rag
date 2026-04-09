"""Elasticsearch learner profile store.

One ES document per learner (index: learner_profiles, keyed by learner_id).
In-memory dict is the primary read source; ES is upserted on every write.

Weighted-average update formula (per skill):
  new_score = (old_score * old_count + contribution) / (old_count + 1)
  new_count = old_count + 1

Session review contributions:
  technically_strong  → +1.0
  technically_weak    → -1.0
  softskills_strong   → +1.0  (tag prefix stripped before storage)
  softskills_weak     → -1.0
  learning_styles     → +1.0  (tag prefix stripped before storage)
"""

import structlog

from src.models.profile_document import LearnerProfileDocument, SkillEntry

logger = structlog.get_logger()

ES_INDEX = "learner_profiles"

# In-memory primary store: learner_id → LearnerProfileDocument
_profiles: dict[str, LearnerProfileDocument] = {}


# ---------------------------------------------------------------------------
# ES client (lazy, same pattern as other stores)
# ---------------------------------------------------------------------------

def _get_es_client():
    try:
        from elasticsearch import AsyncElasticsearch
        from config.settings import settings
        if not settings.elasticsearch_url:
            return None
        return AsyncElasticsearch(settings.elasticsearch_url)
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Weighted-average helper
# ---------------------------------------------------------------------------

def _apply_contribution(
    skill_map: dict[str, SkillEntry],
    key: str,
    contribution: float,
) -> None:
    """Update a single skill entry in-place using weighted average."""
    if key in skill_map:
        old = skill_map[key]
        new_count = old.count + 1
        skill_map[key] = SkillEntry(
            score=(old.score * old.count + contribution) / new_count,
            count=new_count,
        )
    else:
        skill_map[key] = SkillEntry(score=contribution, count=1)


def _strip_prefix(tag: str) -> str:
    """'softskill:decomposition' → 'decomposition'  (handles any single prefix)."""
    return tag.split(":", 1)[1] if ":" in tag else tag


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class LearnerProfileStore:
    """Long-term learner profile backed by Elasticsearch."""

    async def get(self, learner_id: str) -> LearnerProfileDocument:
        """Return profile doc; creates an empty one if not found."""
        if learner_id in _profiles:
            return _profiles[learner_id]

        es = _get_es_client()
        if es:
            try:
                result = await es.get(index=ES_INDEX, id=learner_id)
                doc = LearnerProfileDocument(**result["_source"])
                _profiles[learner_id] = doc
                return doc
            except Exception:
                pass
            finally:
                await es.close()

        return LearnerProfileDocument(learner_id=learner_id)

    async def save(self, doc: LearnerProfileDocument) -> None:
        """Update in-memory store and upsert to ES."""
        _profiles[doc.learner_id] = doc

        es = _get_es_client()
        if not es:
            logger.debug("learner_profile_store.save.in_memory", learner_id=doc.learner_id)
            return
        try:
            await es.index(
                index=ES_INDEX,
                id=doc.learner_id,
                document=doc.model_dump(mode="json"),
            )
            logger.info("learner_profile_store.saved", learner_id=doc.learner_id)
        except Exception as exc:
            logger.error("learner_profile_store.save.failed", learner_id=doc.learner_id, error=str(exc))
        finally:
            await es.close()

    async def update_from_review(
        self,
        learner_id: str,
        session_id: str,
        review_result,   # SessionReviewResult — typed loosely to avoid circular import
    ) -> None:
        """Apply session review output to the learner profile via weighted average.

        Contributions per category:
          technically_strong / softskills_strong / learning_styles → +1.0
          technically_weak   / softskills_weak                      → -1.0
        """
        doc = await self.get(learner_id)

        # Technical skills
        for topic in review_result.technically_strong:
            _apply_contribution(doc.technical_skills, topic, +1.0)
        for topic in review_result.technically_weak:
            _apply_contribution(doc.technical_skills, topic, -1.0)

        # Soft skills (strip "softskill:" prefix)
        for tag in review_result.softskills_strong:
            _apply_contribution(doc.softskills, _strip_prefix(tag), +1.0)
        for tag in review_result.softskills_weak:
            _apply_contribution(doc.softskills, _strip_prefix(tag), -1.0)

        # Learning styles (strip "learnstyle:" prefix)
        for tag in review_result.learning_styles:
            _apply_contribution(doc.learning_style, _strip_prefix(tag), +1.0)

        doc.last_updated_session_id = session_id

        await self.save(doc)
        logger.info(
            "learner_profile_store.updated",
            learner_id=learner_id,
            session_id=session_id,
        )
