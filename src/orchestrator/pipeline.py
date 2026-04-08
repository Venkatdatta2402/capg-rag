"""RAG Pipeline — LangGraph implementation.

Graph:
  fetch_profile → lookup_keywords → transform_query → retrieve
      → build_context → rerank → select_prompt → generate → END

Quiz generation is decided inside `generate` and folded into the same LLM call:
  - Retrieval quality GOOD  AND  not retry_mode  → generate_quiz=True
  - Otherwise                                    → generate_quiz=False (quiz_form.skipped=True)

Expected answers are saved to QuizStore (server-side) before returning.
The client receives only questions — submitting them to POST /quiz/submit
triggers the separate judge model.
"""

import uuid
from typing import Any, TypedDict

import structlog
from langgraph.graph import END, StateGraph

from config.settings import settings
from src.agents.context_builder import ContextObjectBuilder
from src.agents.judge import JudgeAgent
from src.agents.query_transform import QueryTransformAgent
from src.agents.rag_agent import RAGAgent
from src.llm.factory import get_llm_client
from src.models.learner import LearnerProfile, RecentInteraction, SessionState
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
from src.models.interaction import (
    ContextChunk, Interaction, InteractionMeta, QuizData, QuizQuestion as ESQuizQuestion,
)
from src.storage.interaction_store import InteractionStore
from src.storage.session_memory import SessionMemoryStore
from src.storage.user_profile import UserProfileStore
from src.tools.keyword_lookup import invoke_keyword_lookup

logger = structlog.get_logger()


class PipelineState(TypedDict):
    query_input: QueryInput
    profile: LearnerProfile | None
    session: SessionState | None
    domain_keywords: list[str]
    enriched_query: EnrichedQuery | None
    context_obj: ContextObject | None
    candidates: list[RetrievalResult]
    reranked_chunks: list[RerankedChunk]
    prompt_version: Any | None
    response: GenerationResponse | None
    _prompt_override_id: str        # skip selector and use this version directly


class Pipeline(BaseOrchestrator):
    """Sequential RAG pipeline: query transform → hierarchical retrieval → generation."""

    def __init__(self):
        context_llm = get_llm_client(settings.context_provider, settings.context_model)
        generation_llm = get_llm_client(settings.generation_provider, settings.generation_model)
        judge_llm = get_llm_client(settings.judge_provider, settings.judge_model)

        profile_store = UserProfileStore()
        session_store = SessionMemoryStore()
        interaction_store = InteractionStore()

        vector_store = VectorStore()
        hierarchical_retriever = HierarchicalRetriever(vector_store)
        reranker = Reranker()

        query_transform = QueryTransformAgent(llm=context_llm)
        context_builder = ContextObjectBuilder()
        rag_agent = RAGAgent(llm=generation_llm)
        self.judge_agent = JudgeAgent(llm=judge_llm)   # exposed for /quiz/submit

        registry = PromptRegistry()
        selector = PromptSelector(registry, CanaryRouter(), llm=context_llm)

        self._registry = registry
        self._selector = selector
        self._graph = self._build_graph(
            profile_store, session_store, interaction_store,
            query_transform, context_builder,
            hierarchical_retriever, reranker,
            selector, rag_agent, registry,
        )

    @staticmethod
    def _build_graph(
        profile_store, session_store, interaction_store,
        query_transform, context_builder,
        hierarchical_retriever, reranker,
        selector, rag_agent, registry,
    ):
        async def fetch_profile(state: PipelineState) -> dict:
            q = state["query_input"]
            profile = await profile_store.get(q.user_id)
            session = await session_store.get(q.session_id, q.user_id)
            logger.debug("pipeline.fetch_profile.done", grade=profile.grade)
            return {"profile": profile, "session": session}

        async def lookup_keywords(state: PipelineState) -> dict:
            """Fetch curated domain keywords for this learner's grade.

            Subject is not yet known (query not transformed yet), so we look up
            by grade alone. The transform agent selects the most relevant terms.
            """
            profile = state["profile"]
            keywords = await invoke_keyword_lookup(grade=profile.grade, subject="")
            logger.debug("pipeline.lookup_keywords.done", grade=profile.grade, count=len(keywords))
            return {"domain_keywords": keywords}

        async def transform_query(state: PipelineState) -> dict:
            enriched = await query_transform.run(
                state["query_input"],
                state["profile"],
                domain_keywords=state["domain_keywords"],
            )
            logger.debug("pipeline.transform_query.done", keywords=enriched.keywords)
            return {"enriched_query": enriched}

        async def retrieve(state: PipelineState) -> dict:
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
            logger.debug("pipeline.retrieve.done", candidates=len(candidates))
            return {"candidates": candidates}

        async def build_context(state: PipelineState) -> dict:
            session = state["session"]
            if session.context_object is not None:
                # Reuse persisted ContextObject (retry_mode already updated by judge)
                context_obj = session.context_object
                logger.debug("pipeline.build_context.reused", retry_mode=context_obj.retry_mode)
            else:
                # First interaction in this session — build fresh and persist
                context_obj = context_builder.build(state["profile"], session)
                q = state["query_input"]
                await session_store.save_context(q.session_id, q.user_id, context_obj)
                logger.debug("pipeline.build_context.built", retry_mode=context_obj.retry_mode)
            return {"context_obj": context_obj}

        async def rerank_node(state: PipelineState) -> dict:
            reranked = reranker.rerank(
                state["enriched_query"].rewritten_text,
                state["candidates"],
            )
            logger.debug("pipeline.rerank.done", chunks=len(reranked))
            return {"reranked_chunks": reranked}

        async def select_prompt(state: PipelineState) -> dict:
            override_id = state.get("_prompt_override_id", "")
            if override_id:
                version = await registry.get(override_id)
                if version:
                    logger.debug("pipeline.select_prompt.override", version=override_id)
                    return {"prompt_version": version}
            topic = state["enriched_query"].topic if state["enriched_query"] else ""
            prompt_version = await selector.select(state["context_obj"], topic=topic)
            logger.debug("pipeline.select_prompt.done", version=prompt_version.version_id)
            return {"prompt_version": prompt_version}

        async def generate(state: PipelineState) -> dict:
            """Answer + optional MCQ quiz via present_mcq tool call.

            Quiz conditions (both must hold):
              - Retrieval quality GOOD  — answer is reliable enough to test on
              - Not in retry mode       — learner is not already struggling
            """
            context_obj = state["context_obj"]
            chunks = state["reranked_chunks"]
            enriched = state["enriched_query"]
            query_input = state["query_input"]
            profile = state["profile"]
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

            # Build ES quiz questions (include correct_answer for server-side storage)
            es_questions = [
                ESQuizQuestion(
                    question_id=q.question_id,
                    question=q.question,
                    options=q.options,
                    correct_answer=quiz_keys.get(q.question_id, ""),
                )
                for q in response.quiz_form.questions
            ]

            # Derive difficulty from prompt variant
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
                grade=profile.grade,
                interaction=interaction,
            )

            # Append interaction to runtime session window (recent 5 + rolling summary)
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
        logger.info("pipeline.start", query=query_input.query_text[:50])
        initial_state: PipelineState = {
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

        Used by /quiz/regenerate after the learner picks one of the 3 prompt
        MCQ choices. Everything else (re-retrieval, reranking, generation) runs
        normally so fresh context is fetched with the hint-enriched query.
        """
        logger.info(
            "pipeline.start_with_override",
            query=query_input.query_text[:50],
            prompt_override=prompt_version_id,
        )
        initial_state: PipelineState = {
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
