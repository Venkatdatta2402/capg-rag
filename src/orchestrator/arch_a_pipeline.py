"""Architecture A Pipeline — LangGraph implementation.

Graph:
  enrich_and_retrieve → rerank → select_prompt → generate → END

Key characteristic: Context enrichment and raw query retrieval run IN PARALLEL
inside the `enrich_and_retrieve` node using asyncio.gather, then results merge
for reranking. All inter-node data flows through typed state.
"""

import asyncio
from typing import Any, TypedDict

import structlog
from langgraph.graph import END, StateGraph

from config.settings import settings
from src.agents.arch_a.context_rephrase import ContextRephraseAgent
from src.agents.shared.rag_agent import RAGAgent
from src.llm.factory import get_llm_client
from src.models.query import ContextObject, EnrichedQuery, QueryInput
from src.models.response import GenerationResponse
from src.models.retrieval import RerankedChunk, RetrievalResult
from src.orchestrator.base import BaseOrchestrator
from src.prompt_service.canary import CanaryRouter
from src.prompt_service.registry import PromptRegistry
from src.prompt_service.selector import PromptSelector
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.sparse import SparseRetriever
from src.retrieval.vector_store import VectorStore
from src.storage.session_memory import SessionMemoryStore
from src.storage.user_profile import UserProfileStore
from src.tools.keyword_lookup import invoke_keyword_lookup

logger = structlog.get_logger()


class ArchAState(TypedDict):
    """Typed state passed between all nodes in the Architecture A graph."""

    query_input: QueryInput
    _profile_grade: str                 # temp: grade fetched before keyword lookup
    domain_keywords: list[str]          # from keyword store, keyed by grade+subject
    enriched_query: EnrichedQuery | None
    context_obj: ContextObject | None
    raw_results: list[RetrievalResult]
    reranked_chunks: list[RerankedChunk]
    prompt_version: Any | None           # PromptVersion (Any avoids circular at type-check)
    response: GenerationResponse | None


class ArchAPipeline(BaseOrchestrator):
    """Architecture A: Parallel context enrichment + raw query retrieval (LangGraph)."""

    def __init__(self):
        # LLM clients
        context_llm = get_llm_client(settings.context_provider, settings.context_model)
        generation_llm = get_llm_client(settings.generation_provider, settings.generation_model)

        # Storage
        profile_store = UserProfileStore()
        session_store = SessionMemoryStore()

        # Retrieval
        vector_store = VectorStore()
        sparse_retriever = SparseRetriever()
        hybrid_retriever = HybridRetriever(vector_store, sparse_retriever)
        reranker = Reranker()

        # Agents
        context_agent = ContextRephraseAgent(
            llm=context_llm,
            profile_store=profile_store,
            session_store=session_store,
        )
        rag_agent = RAGAgent(llm=generation_llm)

        # Prompt service
        registry = PromptRegistry()
        selector = PromptSelector(registry, CanaryRouter())

        # Build and compile the graph, capturing component refs in closures
        self._graph = self._build_graph(
            context_agent, hybrid_retriever, reranker, selector, rag_agent,
            profile_store,
        )

    # ------------------------------------------------------------------
    # Node definitions
    # ------------------------------------------------------------------

    @staticmethod
    def _build_graph(context_agent, hybrid_retriever, reranker, selector, rag_agent,
                     profile_store):

        async def fetch_profile(state: ArchAState) -> dict:
            """Fetch learner profile so grade is available for keyword lookup."""
            q = state["query_input"]
            profile = await profile_store.get(q.user_id)
            logger.debug("arch_a.fetch_profile.done", grade=profile.grade)
            return {"_profile_grade": profile.grade}

        async def lookup_keywords(state: ArchAState) -> dict:
            """Tool node: fetch domain keywords before context enrichment begins."""
            keywords = await invoke_keyword_lookup(
                grade=state.get("_profile_grade", ""),
                subject="",
            )
            logger.debug("arch_a.lookup_keywords.done", count=len(keywords))
            return {"domain_keywords": keywords}

        async def enrich_and_retrieve(state: ArchAState) -> dict:
            """Run context enrichment and raw query retrieval IN PARALLEL.

            Domain keywords are already in state — passed into the context agent
            so the LLM selects precise NCERT terms when rewriting the query.
            """
            query_input = state["query_input"]

            context_task = asyncio.create_task(
                context_agent.run(query_input, domain_keywords=state["domain_keywords"])
            )
            retrieval_task = asyncio.create_task(
                hybrid_retriever.search(query_input.query_text)
            )
            (enriched_query, context_obj), raw_results = await asyncio.gather(
                context_task, retrieval_task
            )

            logger.debug("arch_a.enrich_and_retrieve.done", results=len(raw_results))
            return {
                "enriched_query": enriched_query,
                "context_obj": context_obj,
                "raw_results": raw_results,
            }

        async def rerank_node(state: ArchAState) -> dict:
            reranked = reranker.rerank(
                state["enriched_query"].rewritten_text,
                state["raw_results"],
            )
            logger.debug("arch_a.rerank.done", chunks=len(reranked))
            return {"reranked_chunks": reranked}

        async def select_prompt(state: ArchAState) -> dict:
            prompt_version = await selector.select(state["context_obj"])
            logger.debug("arch_a.select_prompt.done", version=prompt_version.version_id)
            return {"prompt_version": prompt_version}

        async def generate(state: ArchAState) -> dict:
            response, _ = await rag_agent.run(
                enriched_query=state["enriched_query"],
                prompt_version=state["prompt_version"],
                chunks=state["reranked_chunks"],
                architecture="A",
            )
            logger.info(
                "arch_a.generate.done",
                latency_ms=response.metadata.latency_ms,
                retrieval_quality=response.metadata.retrieval_quality_flag,
            )
            return {"response": response}

        # Graph wiring
        workflow = StateGraph(ArchAState)

        workflow.add_node("fetch_profile", fetch_profile)
        workflow.add_node("lookup_keywords", lookup_keywords)
        workflow.add_node("enrich_and_retrieve", enrich_and_retrieve)
        workflow.add_node("rerank", rerank_node)
        workflow.add_node("select_prompt", select_prompt)
        workflow.add_node("generate", generate)

        workflow.set_entry_point("fetch_profile")
        workflow.add_edge("fetch_profile", "lookup_keywords")
        workflow.add_edge("lookup_keywords", "enrich_and_retrieve")
        workflow.add_edge("enrich_and_retrieve", "rerank")
        workflow.add_edge("rerank", "select_prompt")
        workflow.add_edge("select_prompt", "generate")
        workflow.add_edge("generate", END)

        return workflow.compile()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def run(self, query_input: QueryInput) -> GenerationResponse:
        logger.info("arch_a.start", query=query_input.query_text[:50])

        initial_state: ArchAState = {
            "query_input": query_input,
            "domain_keywords": [],
            "_profile_grade": "",
            "enriched_query": None,
            "context_obj": None,
            "raw_results": [],
            "reranked_chunks": [],
            "prompt_version": None,
            "response": None,
        }

        final_state = await self._graph.ainvoke(initial_state)
        return final_state["response"]
