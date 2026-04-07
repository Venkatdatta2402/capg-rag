"""Governance Pipeline — LangGraph implementation of Loop B.

Graph:
  gather_feedback
      → [has_records?] → END (skip) | analyze
      → assess_risk
      → [proceed?] → skip_version | suggest
      → [candidate?] → skip_version | experiment
      → [passed?] → reject_candidate | deploy_candidate
      → next_version
      → [more_versions?] → assess_risk (loop) | END

LangGraph conditional edges make the skip/proceed/loop logic explicit and auditable.
"""

from typing import TypedDict

import structlog
from langgraph.graph import END, StateGraph

from src.governance.analysis import AnalysisAgent, AnalysisResult
from src.governance.experiment import ExperimentAgent, ExperimentResult
from src.governance.risk import RiskAgent, RiskAssessment
from src.governance.suggestion import SuggestionAgent, SuggestionResult
from src.models.prompt import PromptVersion
from src.prompt_service.registry import PromptRegistry
from src.storage.feedback_store import FeedbackStore

logger = structlog.get_logger()


class GovernanceState(TypedDict):
    """Typed state for the governance pipeline graph."""

    # Inputs
    records: list                       # FeedbackRecord list from the Feedback Store

    # Analysis output
    analysis: AnalysisResult | None
    weak_version_ids: list[str]         # Queue of prompt versions to process

    # Per-version working state (updated each loop iteration)
    current_version_id: str
    current_prompt: PromptVersion | None
    risk: RiskAssessment | None
    suggestion: SuggestionResult | None
    experiment_result: ExperimentResult | None

    # Accumulated results across all versions
    actions: list[dict]


class GovernancePipeline:
    """Orchestrates the offline prompt improvement loop (Loop B) using LangGraph.

    Activation triggers:
    - Scheduled batch job (e.g. nightly)
    - Feedback volume threshold crossed (e.g. 500 new records)
    - Manual trigger by developer
    """

    def __init__(self, feedback_store: FeedbackStore, prompt_registry: PromptRegistry):
        self._feedback = feedback_store
        self._registry = prompt_registry

        analysis_agent = AnalysisAgent()
        risk_agent = RiskAgent()
        experiment_agent = ExperimentAgent()
        suggestion_agent = SuggestionAgent()

        self._graph = self._build_graph(
            feedback_store, prompt_registry,
            analysis_agent, risk_agent, experiment_agent, suggestion_agent,
        )

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_graph(
        feedback_store, prompt_registry,
        analysis_agent, risk_agent, experiment_agent, suggestion_agent,
    ):
        # ── Nodes ──────────────────────────────────────────────────────

        async def gather_feedback(state: GovernanceState) -> dict:
            records = await feedback_store.get_records_since_last_governance_run()
            logger.info("governance.gather_feedback", count=len(records))
            return {"records": records}

        async def analyze(state: GovernanceState) -> dict:
            analysis = await analysis_agent.analyze(state["records"])
            logger.info(
                "governance.analyze",
                prompt_failure_rate=analysis.prompt_failure_rate,
                weak_versions=analysis.weak_prompt_versions,
            )
            return {
                "analysis": analysis,
                "weak_version_ids": list(analysis.weak_prompt_versions),
                "actions": [],
            }

        async def assess_risk(state: GovernanceState) -> dict:
            """Load the next version and assess its risk."""
            version_id = state["weak_version_ids"][0]
            prompt = await prompt_registry.get(version_id)
            if not prompt:
                logger.warning("governance.assess_risk.not_found", version_id=version_id)
                return {
                    "current_version_id": version_id,
                    "current_prompt": None,
                    "risk": None,
                }
            risk = await risk_agent.assess(prompt, state["analysis"])
            logger.info(
                "governance.assess_risk",
                version_id=version_id,
                risk_level=risk.risk_level,
            )
            return {
                "current_version_id": version_id,
                "current_prompt": prompt,
                "risk": risk,
            }

        async def suggest(state: GovernanceState) -> dict:
            suggestion = await suggestion_agent.suggest(
                state["current_prompt"], state["analysis"], state["risk"]
            )
            logger.info(
                "governance.suggest",
                skipped=suggestion.skipped,
                candidate=suggestion.candidate.candidate_id if suggestion.candidate else None,
            )
            return {"suggestion": suggestion}

        async def run_experiment(state: GovernanceState) -> dict:
            result = await experiment_agent.run_experiment(
                state["suggestion"].candidate,
                baseline_score=state["analysis"].prompt_failure_rate,
            )
            logger.info(
                "governance.experiment",
                candidate_id=result.candidate_id,
                passed=result.passed,
            )
            return {"experiment_result": result}

        async def deploy_candidate(state: GovernanceState) -> dict:
            candidate = state["suggestion"].candidate
            await prompt_registry.add_candidate(candidate)
            action = {
                "version": state["current_version_id"],
                "action": "candidate_deployed",
                "candidate_id": candidate.candidate_id,
            }
            logger.info("governance.deploy", **action)
            return {"actions": state["actions"] + [action]}

        async def reject_candidate(state: GovernanceState) -> dict:
            action = {
                "version": state["current_version_id"],
                "action": "experiment_failed",
                "candidate_id": state["suggestion"].candidate.candidate_id,
            }
            logger.info("governance.reject", **action)
            return {"actions": state["actions"] + [action]}

        async def skip_version(state: GovernanceState) -> dict:
            reason = (
                state["suggestion"].skip_reason
                if state.get("suggestion") and state["suggestion"].skipped
                else f"Risk too high ({state['risk'].risk_level})" if state.get("risk") else "Prompt not found"
            )
            action = {
                "version": state["current_version_id"],
                "action": "skipped",
                "reason": reason,
            }
            logger.info("governance.skip", **action)
            return {"actions": state["actions"] + [action]}

        async def next_version(state: GovernanceState) -> dict:
            """Pop the processed version from the queue."""
            remaining = state["weak_version_ids"][1:]
            return {
                "weak_version_ids": remaining,
                "current_version_id": "",
                "current_prompt": None,
                "risk": None,
                "suggestion": None,
                "experiment_result": None,
            }

        # ── Conditional edge functions ──────────────────────────────────

        def route_after_gather(state: GovernanceState) -> str:
            """Skip the whole pipeline if there are no feedback records."""
            return "analyze" if state["records"] else END

        def route_after_analyze(state: GovernanceState) -> str:
            """Skip if there are no weak prompt versions to improve."""
            return "assess_risk" if state["weak_version_ids"] else END

        def route_after_risk(state: GovernanceState) -> str:
            """Skip version if prompt not found or risk is too high."""
            if not state["current_prompt"]:
                return "skip_version"
            return "suggest" if state["risk"].proceed else "skip_version"

        def route_after_suggest(state: GovernanceState) -> str:
            """Skip version if suggestion was skipped or produced no candidate."""
            s = state.get("suggestion")
            if not s or s.skipped or not s.candidate:
                return "skip_version"
            return "run_experiment"

        def route_after_experiment(state: GovernanceState) -> str:
            """Deploy if experiment passed, reject otherwise."""
            return "deploy_candidate" if state["experiment_result"].passed else "reject_candidate"

        def route_after_next_version(state: GovernanceState) -> str:
            """Loop back if more versions remain, otherwise end."""
            return "assess_risk" if state["weak_version_ids"] else END

        # ── Graph wiring ───────────────────────────────────────────────

        workflow = StateGraph(GovernanceState)

        workflow.add_node("gather_feedback", gather_feedback)
        workflow.add_node("analyze", analyze)
        workflow.add_node("assess_risk", assess_risk)
        workflow.add_node("suggest", suggest)
        workflow.add_node("run_experiment", run_experiment)
        workflow.add_node("deploy_candidate", deploy_candidate)
        workflow.add_node("reject_candidate", reject_candidate)
        workflow.add_node("skip_version", skip_version)
        workflow.add_node("next_version", next_version)

        workflow.set_entry_point("gather_feedback")

        workflow.add_conditional_edges("gather_feedback", route_after_gather)
        workflow.add_conditional_edges("analyze", route_after_analyze)
        workflow.add_conditional_edges("assess_risk", route_after_risk)
        workflow.add_conditional_edges("suggest", route_after_suggest)
        workflow.add_conditional_edges("run_experiment", route_after_experiment)

        # Both deploy and reject flow to next_version
        workflow.add_edge("deploy_candidate", "next_version")
        workflow.add_edge("reject_candidate", "next_version")
        workflow.add_edge("skip_version", "next_version")

        # Loop or end
        workflow.add_conditional_edges("next_version", route_after_next_version)

        return workflow.compile()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def run(self) -> dict:
        """Execute the full governance pipeline."""
        logger.info("governance.pipeline.start")

        initial_state: GovernanceState = {
            "records": [],
            "analysis": None,
            "weak_version_ids": [],
            "current_version_id": "",
            "current_prompt": None,
            "risk": None,
            "suggestion": None,
            "experiment_result": None,
            "actions": [],
        }

        final_state = await self._graph.ainvoke(initial_state)

        analysis = final_state.get("analysis")
        summary = {
            "status": "completed",
            "records_analyzed": analysis.total_records if analysis else 0,
            "prompt_failure_rate": analysis.prompt_failure_rate if analysis else 0,
            "retrieval_failure_rate": analysis.retrieval_failure_rate if analysis else 0,
            "actions": final_state.get("actions", []),
        }

        logger.info("governance.pipeline.done", **summary)
        return summary
