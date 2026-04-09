"""RAG Pipeline — LangGraph implementation.

Graph:
  fetch_profile → initial_retrieve → transform_query ┬→ retrieve → rerank ┐
                                                      └→ select_prompt ───┘→ generate → END

Retrieval is two-pass:
  Pass 1 (initial_retrieve): Hybrid (dense+sparse), grade-scoped only — broad signal collection
    → chunk keywords/concepts fed into transform_query
  Pass 2 (retrieve): Hierarchical+hybrid, enriched+rewritten query — precise scoped retrieval

select_prompt forks from transform_query in parallel with the retrieval chain.
"""

import uuid
from typing import Any, TypedDict

import structlog
from langgraph.graph import END, StateGraph

from config.settings import settings
from src.agents.judge import JudgeAgent
from src.agents.query_transform import QueryTransformAgent
from src.agents.rag_agent import RAGAgent
from src.llm.factory import get_llm_client
from src.models.learner import RecentInteraction, SessionState
from src.models.profile_document import LearnerProfileDocument
from src.models.query import ContextObject, EnrichedQuery, QueryInput
from src.models.response import GenerationResponse
from src.models.retrieval import RerankedChunk, RetrievalResult
from src.orchestrator.base import BaseOrchestrator
from src.prompt_service.canary import CanaryRouter
from src.prompt_service.registry import PromptRegistry
from src.prompt_service.selector import PromptSelector
from src.retrieval.hierarchical import HierarchicalRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.sparse import SparseRetriever
from src.retrieval.vector_store import VectorStore
from src.models.interaction import (
    ContextChunk, Interaction, InteractionMeta, QuizData, QuizQuestion as ESQuizQuestion,
)
from src.storage.interaction_store import InteractionStore
from src.storage.learner_profile_store import LearnerProfileStore
from src.storage.session_memory import SessionMemoryStore

logger = structlog.get_logger()


class PipelineState(TypedDict):
    query_input: QueryInput
    profile_doc: LearnerProfileDocument | None
    session: SessionState | None
    initial_candidates: list[RetrievalResult]     # pass-1: broad hybrid, signals only
    enriched_query: EnrichedQuery | None          # output of transform_query (rewritten + selected signals)
    candidates: list[RetrievalResult]             # pass-2: hierarchical+hybrid, scoped
    reranked_chunks: list[RerankedChunk]
    context_obj: ContextObject | None
    prompt_version: Any | None
    response: GenerationResponse | None
    _judge_followup: bool                         # True → skip session context in transform_query


class Pipeline(BaseOrchestrator):
    """Two-pass retrieval RAG pipeline with chunk-signal-driven query transformation."""

    def __init__(self):
        context_llm = get_llm_client(settings.context_provider, settings.context_model)
        generation_llm = get_llm_client(settings.generation_provider, settings.generation_model)
        judge_llm = get_llm_client(settings.judge_provider, settings.judge_model)

        profile_store = LearnerProfileStore()
        session_store = SessionMemoryStore()
        interaction_store = InteractionStore()

        vector_store = VectorStore()
        sparse_retriever = SparseRetriever()
        hybrid_retriever = HybridRetriever(vector_store, sparse_retriever)
        hierarchical_retriever = HierarchicalRetriever(vector_store, sparse_retriever)
        reranker = Reranker()

        query_transform = QueryTransformAgent(llm=context_llm)
        rag_agent = RAGAgent(llm=generation_llm)
        self.judge_agent = JudgeAgent(llm=judge_llm)

        registry = PromptRegistry()
        selector = PromptSelector(registry, CanaryRouter(), llm=context_llm)

        self._registry = registry
        self._selector = selector
        self._graph = self._build_graph(
            profile_store, session_store, interaction_store,
            query_transform, hybrid_retriever, hierarchical_retriever,
            reranker, selector, rag_agent,
        )

    @staticmethod
    def _build_graph(
        profile_store, session_store, interaction_store,
        query_transform, hybrid_retriever, hierarchical_retriever,
        reranker, selector, rag_agent,
    ):
        async def fetch_profile(state: PipelineState) -> dict:
            q = state["query_input"]
            profile_doc = await profile_store.get(q.user_id)
            session = await session_store.get(q.session_id, q.user_id)
            logger.debug("pipeline.fetch_profile.done", grade=profile_doc.grade)
            return {"profile_doc": profile_doc, "session": session}

        async def initial_retrieve(state: PipelineState) -> dict:
            """Pass 1: broad hybrid retrieval, grade-scoped only.

            Uses the raw query text — subject not yet known.
            Purpose is signal collection (chunk keywords + concepts),
            not final answer retrieval.
            """
            candidates = await hybrid_retriever.search(
                query=state["query_input"].query_text,
                grade=state["profile_doc"].grade,
                top_k=settings.top_k_retrieval,
            )
            logger.debug("pipeline.initial_retrieve.done", candidates=len(candidates))
            return {"initial_candidates": candidates}

        async def transform_query(state: PipelineState) -> dict:
            """Rewrite query using chunk signals from pass-1.

            Pools keywords + concepts from initial_candidates and passes to the
            transform agent, which selects relevant ones and rewrites the query.
            Session context added for disambiguation (skipped on judge follow-up).
            """
            session_context = ""
            if not state.get("_judge_followup"):
                session = state.get("session")
                if session:
                    parts = []
                    past = session.summary_of_past
                    if past.covered_topics:
                        parts.append(f"- Covered topics: {', '.join(past.covered_topics)}")
                    if past.performance_trend != "unknown":
                        parts.append(f"- Performance trend: {past.performance_trend}")
                    if session.recent_interactions:
                        recent = [ix.topic for ix in session.recent_interactions if ix.topic]
                        if recent:
                            parts.append(f"- Recent topics: {', '.join(recent[-3:])}")
                    session_context = "\n".join(parts)

            enriched = await query_transform.run(
                state["query_input"],
                grade=state["profile_doc"].grade,
                initial_chunks=state["initial_candidates"],
                session_context=session_context,
            )
            logger.debug("pipeline.transform_query.done", keywords=enriched.keywords)
            return {"enriched_query": enriched}

        async def retrieve(state: PipelineState) -> dict:
            """Pass 2: hard hierarchical retrieval using the enriched query.

            Resolves unit → chapter → section → subsection step-by-step,
            then does dense+sparse RRF within the narrowest resolved scope.
            """
            enriched = state["enriched_query"]
            candidates = await hierarchical_retriever.search(
                query=enriched.rewritten_text,
                keywords=enriched.keywords,
                grade=state["profile_doc"].grade,
                subject=enriched.subject,
            )
            logger.debug("pipeline.retrieve.done", candidates=len(candidates))
            return {"candidates": candidates}

        async def rerank_node(state: PipelineState) -> dict:
            reranked = reranker.rerank(
                state["enriched_query"].rewritten_text,
                state["candidates"],
            )
            logger.debug("pipeline.rerank.done", chunks=len(reranked))
            return {"reranked_chunks": reranked}

        async def select_prompt(state: PipelineState) -> dict:
            q = state["query_input"]
            existing_ctx = state["session"].context_object
            retry_mode = existing_ctx.retry_mode if existing_ctx else False
            retry_count = existing_ctx.retry_count if existing_ctx else 0

            prompt_version, context_obj = await selector.select(
                profile_doc=state["profile_doc"],
                enriched=state["enriched_query"],
                retry_mode=retry_mode,
                retry_count=retry_count,
            )
            await session_store.save_context(q.session_id, q.user_id, context_obj)
            logger.debug("pipeline.select_prompt.done", version=prompt_version.version_id)
            return {"prompt_version": prompt_version, "context_obj": context_obj}

        async def generate(state: PipelineState) -> dict:
            context_obj = state["context_obj"]
            chunks = state["reranked_chunks"]
            enriched = state["enriched_query"]
            query_input = state["query_input"]
            prompt_version = state["prompt_version"]

            avg_score = sum(c.rerank_score for c in chunks) / len(chunks) if chunks else 0.0
            generate_quiz = (avg_score > 0.75) and (not context_obj.retry_mode)

            interaction_id = str(uuid.uuid4())
            quiz_id = query_input.session_id

            response, quiz_keys = await rag_agent.run(
                enriched_query=enriched,
                prompt_version=prompt_version,
                chunks=chunks,
                generate_quiz=generate_quiz,
                quiz_id=quiz_id,
            )

            if not generate_quiz:
                reason = (
                    "retry_mode=True" if context_obj.retry_mode
                    else f"retrieval_quality={'GOOD' if avg_score > 0.75 else 'MARGINAL/POOR'}"
                )
                response.quiz_form.skipped = True
                response.quiz_form.skip_reason = reason
                logger.info("pipeline.quiz.skipped", reason=reason)
            else:
                logger.info(
                    "pipeline.quiz.generated",
                    interaction_id=interaction_id,
                    question_count=len(response.quiz_form.questions),
                )

            es_questions = [
                ESQuizQuestion(
                    question_id=q.question_id,
                    question=q.question,
                    options=q.options,
                    correct_answer=quiz_keys.get(q.question_id, ""),
                )
                for q in response.quiz_form.questions
            ]

            difficulty_map = {"remedial": "easy", "standard": "medium", "advanced": "hard"}
            difficulty = difficulty_map.get(prompt_version.variant, "medium")

            interaction = Interaction(
                interaction_id=interaction_id,
                question=query_input.query_text,
                model_answer=response.answer_text,
                quiz=QuizData(
                    quiz_id=quiz_id,
                    questions=es_questions,
                    status="ignored",
                ),
                context_used=[
                    ContextChunk(chunk_id=c.chunk_id, text=c.text, rank=i + 1)
                    for i, c in enumerate(chunks)
                ],
                meta=InteractionMeta(
                    subject=enriched.subject,
                    topic=enriched.topic,
                    difficulty=difficulty,
                ),
            )

            await interaction_store.append_interaction(
                session_id=query_input.session_id,
                user_id=query_input.user_id,
                grade=state["profile_doc"].grade,
                interaction=interaction,
            )

            recent_ix = RecentInteraction(
                interaction_id=interaction_id,
                question=query_input.query_text,
                model_answer=response.answer_text,
                topic=enriched.topic,
                prompt_version=response.metadata.prompt_version,
                retry_count=context_obj.retry_count,
                quiz_status="ignored",
                score=0,
            )
            await session_store.append_interaction(
                query_input.session_id,
                query_input.user_id,
                recent_ix,
            )

            response.interaction_id = interaction_id
            logger.info(
                "pipeline.generate.done",
                interaction_id=interaction_id,
                latency_ms=response.metadata.latency_ms,
                retrieval_quality=response.metadata.retrieval_quality_flag,
            )
            return {"response": response}

        workflow = StateGraph(PipelineState)
        workflow.add_node("fetch_profile", fetch_profile)
        workflow.add_node("initial_retrieve", initial_retrieve)
        workflow.add_node("transform_query", transform_query)
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("rerank", rerank_node)
        workflow.add_node("select_prompt", select_prompt)
        workflow.add_node("generate", generate)

        workflow.set_entry_point("fetch_profile")
        workflow.add_edge("fetch_profile", "initial_retrieve")
        workflow.add_edge("initial_retrieve", "transform_query")
        # select_prompt and retrieve fork from transform_query in parallel
        workflow.add_edge("transform_query", "retrieve")
        workflow.add_edge("transform_query", "select_prompt")
        workflow.add_edge("retrieve", "rerank")
        # generate waits for both branches
        workflow.add_edge("rerank", "generate")
        workflow.add_edge("select_prompt", "generate")
        workflow.add_edge("generate", END)

        return workflow.compile()

    def _make_initial_state(self, query_input: QueryInput, judge_followup: bool) -> PipelineState:
        return {
            "query_input": query_input,
            "profile_doc": None,
            "session": None,
            "initial_candidates": [],
            "enriched_query": None,
            "candidates": [],
            "reranked_chunks": [],
            "context_obj": None,
            "prompt_version": None,
            "response": None,
            "_judge_followup": judge_followup,
        }

    async def run(self, query_input: QueryInput) -> GenerationResponse:
        logger.info("pipeline.start", query=query_input.query_text[:50])
        final_state = await self._graph.ainvoke(self._make_initial_state(query_input, False))
        return final_state["response"]

    async def run_judge_followup(self, query_input: QueryInput) -> GenerationResponse:
        """Re-run with a judge-generated question. Skips session context in transform_query."""
        logger.info("pipeline.start_judge_followup", query=query_input.query_text[:50])
        final_state = await self._graph.ainvoke(self._make_initial_state(query_input, True))
        return final_state["response"]
