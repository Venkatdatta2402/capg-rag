"""Architecture B Pipeline — LangGraph implementation.

Graph:
  fetch_profile → transform_query → retrieve → build_context → rerank → select_prompt → generate → END

Quiz generation is decided inside `generate` and folded into the same LLM call:
  - Retrieval quality GOOD  AND  not retry_mode  → generate_quiz=True
  - Otherwise                                    → generate_quiz=False  (quiz_form.skipped=True)

Expected answers from the quiz are saved to QuizStore (server-side) before returning.
The client receives only questions in quiz_form — submitting them to POST /quiz/submit
triggers the separate judge model.
"""

import uuid
from typing import Any, TypedDict

import structlog
from langgraph.graph import END, StateGraph

from config.settings import settings
from src.agents.arch_b.context_builder import ContextObjectBuilder
from src.agents.arch_b.judge import SeparateJudgeAgent
from src.agents.arch_b.query_transform import QueryTransformAgent
from src.agents.shared.rag_agent import RAGAgent
from src.llm.factory import get_llm_client
from src.models.learner import LearnerProfile, SessionState
from src.models.query import ContextObject, EnrichedQuery, QueryInput
from src.models.response import GenerationResponse
from src.models.retrieval import RerankedChunk, RetrievalResult
from src.orchestrator.base import BaseOrchestrator
from src.prompt_service.canary import CanaryRouter
from src.prompt_service.registry import PromptRegistry
from src.prompt_service.selector import PromptSelector
from src.retrieval.hierarchical import HierarchicalRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.vector_store import VectorStore
from src.storage.quiz_store import QuizStore
from src.storage.session_memory import SessionMemoryStore
from src.storage.user_profile import UserProfileStore
from src.tools.keyword_lookup import invoke_keyword_lookup

logger = structlog.get_logger()


class ArchBState(TypedDict):
    query_input: QueryInput
    profile: LearnerProfile | None
    session: SessionState | None
    domain_keywords: list[str]          # from keyword store, keyed by grade+subject
    enriched_query: EnrichedQuery | None
    context_obj: ContextObject | None
    candidates: list[RetrievalResult]
    reranked_chunks: list[RerankedChunk]
    prompt_version: Any | None
    response: GenerationResponse | None
    _prompt_override_id: str            # optional: skip selector and use this version directly


class ArchBPipeline(BaseOrchestrator):
    """Architecture B: Sequential query transform → hierarchical retrieval (LangGraph)."""

    def __init__(self):
        context_llm = get_llm_client(settings.context_provider, settings.context_model)
        generation_llm = get_llm_client(settings.generation_provider, settings.generation_model)
        judge_llm = get_llm_client(settings.judge_provider, settings.judge_model)

        profile_store = UserProfileStore()
        session_store = SessionMemoryStore()
        quiz_store = QuizStore()

        vector_store = VectorStore()
        hierarchical_retriever = HierarchicalRetriever(vector_store)
        reranker = Reranker()

        query_transform = QueryTransformAgent(llm=context_llm)
        context_builder = ContextObjectBuilder()
        rag_agent = RAGAgent(llm=generation_llm)
        self.judge_agent = SeparateJudgeAgent(llm=judge_llm)   # exposed for /quiz/submit

        registry = PromptRegistry()
        selector = PromptSelector(registry, CanaryRouter())

        self._registry = registry
        self._selector = selector
        self._graph = self._build_graph(
            profile_store, session_store, quiz_store,
            query_transform, context_builder,
            hierarchical_retriever, reranker,
            selector, rag_agent, registry,
        )

    @staticmethod
    def _build_graph(
        profile_store, session_store, quiz_store,
        query_transform, context_builder,
        hierarchical_retriever, reranker,
        selector, rag_agent, registry,
    ):
        async def fetch_profile(state: ArchBState) -> dict:
            q = state["query_input"]
            profile = await profile_store.get(q.user_id)
            session = await session_store.get(q.session_id, q.user_id)
            logger.debug("arch_b.fetch_profile.done", grade=profile.grade)
            return {"profile": profile, "session": session}

        async def lookup_keywords(state: ArchBState) -> dict:
            """Tool node: fetch curated domain keywords for this learner's grade+subject.

            Subject is not yet known (query not transformed), so we look up all
            subjects registered for this grade and merge. The transform agent will
            select the most relevant terms.
            """
            profile = state["profile"]
            # Look up by grade alone first (subject detected by transform agent)
            # In practice, caller can pre-register per grade+subject combos
            keywords = await invoke_keyword_lookup(
                grade=profile.grade,
                subject="",   # broad lookup; transform agent filters to subject
            )
            logger.debug(
                "arch_b.lookup_keywords.done",
                grade=profile.grade,
                count=len(keywords),
            )
            return {"domain_keywords": keywords}

        async def transform_query(state: ArchBState) -> dict:
            enriched = await query_transform.run(
                state["query_input"],
                state["profile"],
                domain_keywords=state["domain_keywords"],
            )
            logger.debug("arch_b.transform_query.done", keywords=enriched.keywords)
            return {"enriched_query": enriched}

        async def retrieve(state: ArchBState) -> dict:
            enriched = state["enriched_query"]
            profile = state["profile"]
            candidates = await hierarchical_retriever.search(
                query=enriched.rewritten_text,
                keywords=enriched.keywords,
                grade=profile.grade,
                subject=enriched.subject,
                chapter_title=enriched.topic,
                section_number=enriched.sub_topic,
            )
            logger.debug("arch_b.retrieve.done", candidates=len(candidates))
            return {"candidates": candidates}

        async def build_context(state: ArchBState) -> dict:
            context_obj = context_builder.build(
                state["profile"], state["session"], state["enriched_query"]
            )
            logger.debug("arch_b.build_context.done", retry_mode=context_obj.retry_mode)
            return {"context_obj": context_obj}

        async def rerank_node(state: ArchBState) -> dict:
            reranked = reranker.rerank(
                state["enriched_query"].rewritten_text,
                state["candidates"],
            )
            logger.debug("arch_b.rerank.done", chunks=len(reranked))
            return {"reranked_chunks": reranked}

        async def select_prompt(state: ArchBState) -> dict:
            override_id = state.get("_prompt_override_id", "")
            if override_id:
                version = await registry.get(override_id)
                if version:
                    logger.debug("arch_b.select_prompt.override", version=override_id)
                    return {"prompt_version": version}
            prompt_version = await selector.select(state["context_obj"])
            logger.debug("arch_b.select_prompt.done", version=prompt_version.version_id)
            return {"prompt_version": prompt_version}

        async def generate(state: ArchBState) -> dict:
            """Answer + optional quiz in ONE LLM call.

            Quiz conditions (both must hold):
              - Retrieval quality is GOOD  — answer is reliable enough to test on
              - Not in retry mode          — learner is not already struggling
            """
            context_obj = state["context_obj"]
            retrieval_good = True   # determined after run(); checked via quality_flag below

            # We decide generate_quiz after seeing quality — so we optimistically try,
            # then gate on the flag returned in metadata.
            # More precisely: compute avg_score here to decide before the call.
            chunks = state["reranked_chunks"]
            avg_score = sum(c.rerank_score for c in chunks) / len(chunks) if chunks else 0.0
            quality_good = avg_score > 0.75
            not_retrying = not context_obj.retry_mode

            generate_quiz = quality_good and not_retrying
            quiz_id = state["query_input"].session_id   # stable identifier for this interaction

            response, expected_answers = await rag_agent.run(
                enriched_query=state["enriched_query"],
                prompt_version=state["prompt_version"],
                chunks=chunks,
                architecture="B",
                generate_quiz=generate_quiz,
                quiz_id=quiz_id,
            )

            # Store expected answers server-side if quiz was generated
            if generate_quiz and expected_answers:
                await quiz_store.save(
                    quiz_id=quiz_id,
                    explanation=response.answer_text,
                    topic=state["enriched_query"].topic,
                    grade=state["profile"].grade,
                    expected_answers=expected_answers,
                )

            if not generate_quiz:
                reason = (
                    "retry_mode=True" if context_obj.retry_mode
                    else f"retrieval_quality={'GOOD' if quality_good else 'MARGINAL/POOR'}"
                )
                response.quiz_form.skipped = True
                response.quiz_form.skip_reason = reason
                logger.info("arch_b.quiz.skipped", reason=reason)
            else:
                logger.info(
                    "arch_b.quiz.generated",
                    quiz_id=quiz_id,
                    question_count=len(response.quiz_form.questions),
                )

            logger.info(
                "arch_b.generate.done",
                latency_ms=response.metadata.latency_ms,
                retrieval_quality=response.metadata.retrieval_quality_flag,
            )
            return {"response": response}

        # Graph wiring
        workflow = StateGraph(ArchBState)

        workflow.add_node("fetch_profile", fetch_profile)
        workflow.add_node("lookup_keywords", lookup_keywords)
        workflow.add_node("transform_query", transform_query)
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("build_context", build_context)
        workflow.add_node("rerank", rerank_node)
        workflow.add_node("select_prompt", select_prompt)
        workflow.add_node("generate", generate)

        workflow.set_entry_point("fetch_profile")
        workflow.add_edge("fetch_profile", "lookup_keywords")
        workflow.add_edge("lookup_keywords", "transform_query")
        workflow.add_edge("transform_query", "retrieve")
        workflow.add_edge("retrieve", "build_context")
        workflow.add_edge("build_context", "rerank")
        workflow.add_edge("rerank", "select_prompt")
        workflow.add_edge("select_prompt", "generate")
        workflow.add_edge("generate", END)

        return workflow.compile()

    async def run(self, query_input: QueryInput) -> GenerationResponse:
        logger.info("arch_b.start", query=query_input.query_text[:50])

        initial_state: ArchBState = {
            "query_input": query_input,
            "profile": None,
            "session": None,
            "domain_keywords": [],
            "enriched_query": None,
            "context_obj": None,
            "candidates": [],
            "reranked_chunks": [],
            "prompt_version": None,
            "response": None,
            "_prompt_override_id": "",
        }

        final_state = await self._graph.ainvoke(initial_state)
        return final_state["response"]

    async def run_with_prompt_override(
        self,
        query_input: QueryInput,
        prompt_version_id: str,
    ) -> GenerationResponse:
        """Re-run the full pipeline, bypassing prompt selection.

        Used by /quiz/regenerate after the learner picks one of the 3
        prompt MCQ choices returned by the judge evaluation. The rest of
        the pipeline (re-retrieval, re-ranking, generation) runs normally
        so fresh context is fetched with the (potentially hint-enriched) query.
        """
        logger.info(
            "arch_b.start_with_override",
            query=query_input.query_text[:50],
            prompt_override=prompt_version_id,
        )

        initial_state: ArchBState = {
            "query_input": query_input,
            "profile": None,
            "session": None,
            "domain_keywords": [],
            "enriched_query": None,
            "context_obj": None,
            "candidates": [],
            "reranked_chunks": [],
            "prompt_version": None,
            "response": None,
            "_prompt_override_id": prompt_version_id,
        }

        final_state = await self._graph.ainvoke(initial_state)
        return final_state["response"]
