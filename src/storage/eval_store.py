"""Elasticsearch evaluation store.

One document per interaction evaluation, keyed by {session_id}_{interaction_id}.
Written at session end by EvalAgent (fire-and-forget).

In-memory fallback used when ES is unavailable.
"""

import structlog

from src.models.eval import EvalResult

logger = structlog.get_logger()

ES_INDEX = "session_evaluations"

_store: dict[str, EvalResult] = {}


def _get_es_client():
    try:
        from elasticsearch import AsyncElasticsearch
        from config.settings import settings
        if not settings.elasticsearch_url:
            return None
        return AsyncElasticsearch(settings.elasticsearch_url)
    except ImportError:
        return None


class EvalStore:
    """Write-only evaluation result store backed by Elasticsearch."""

    async def write(self, result: EvalResult) -> None:
        doc_id = f"{result.session_id}_{result.interaction.interaction_id}"
        _store[doc_id] = result

        es = _get_es_client()
        if not es:
            logger.debug("eval_store.write.in_memory", doc_id=doc_id)
            return
        try:
            await es.index(
                index=ES_INDEX,
                id=doc_id,
                document=result.model_dump(mode="json"),
            )
            logger.info(
                "eval_store.written",
                session_id=result.session_id,
                interaction_id=result.interaction.interaction_id,
            )
        except Exception as exc:
            logger.error("eval_store.write.failed", doc_id=doc_id, error=str(exc))
        finally:
            await es.close()

    async def get(self, session_id: str, interaction_id: str) -> EvalResult | None:
        doc_id = f"{session_id}_{interaction_id}"
        doc = _store.get(doc_id)
        if doc:
            return doc

        es = _get_es_client()
        if not es:
            return None
        try:
            result = await es.get(index=ES_INDEX, id=doc_id)
            return EvalResult(**result["_source"])
        except Exception:
            return None
        finally:
            await es.close()
