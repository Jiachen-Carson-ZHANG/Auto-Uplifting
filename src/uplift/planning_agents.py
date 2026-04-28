"""Phase II — Experiment Planning Agents.

Agents (in pipeline order):
    CaseRetrievalAgent            — reads ledger, surfaces prior evidence
    HypothesisReasoningAgent      — validate | refute | propose next hypothesis
    UpliftStrategySelectionAgent  — pick learner family + base estimator
    TrialSpecWriterAgent          — produce fully resolved trial plan

Adapted from Erica's experiment_planning.py to use:
    - repo's UpliftLedger + UpliftExperimentRecord (not a custom JSONL schema)
    - repo's UpliftHypothesisStore for hypothesis lifecycle
    - repo's REGISTERED_UPLIFT_TEMPLATES for safe learner-family validation
    - ChatLLM = Callable[[str, str], str] from llm_client.py
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Callable, Optional

from src.models.uplift import (
    UpliftExperimentRecord,
    UpliftHypothesis,
    UpliftHypothesisStatus,
)
from src.uplift.ledger import UpliftLedger
from src.uplift.hypotheses import UpliftHypothesisStore, transition_hypothesis
from src.uplift.templates import REGISTERED_UPLIFT_TEMPLATES
from src.uplift.llm_client import ChatLLM

_SKILLS_DIR = Path(__file__).parent / "skills"


def _load_skill(name: str) -> str:
    path = _SKILLS_DIR / f"{name}.md"
    text = path.read_text(encoding="utf-8")
    match = re.search(r"## System Prompt\s*\n+```[^\n]*\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def _parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {}


# ── Intermediate data structures (transient, not stored) ─────────────────────

@dataclass
class RetrievedContext:
    similar_recipes: list[dict]
    supported_hypotheses: list[str]
    refuted_hypotheses: list[str]
    best_learner_family: str
    failed_runs: list[dict]
    summary: str


@dataclass
class HypothesisDecision:
    action: str        # "validate" | "refute" | "propose"
    hypothesis: str
    evidence: str
    confidence: float  # 0–1


@dataclass
class UpliftStrategy:
    learner_family: str
    base_estimator: str
    feature_recipe: str
    split_seed: int
    eval_cutoff: float
    rationale: str


@dataclass
class PlanningTrialSpec:
    """Planning-phase trial spec.  Converted to UpliftTrialSpec after features are built."""
    trial_id: str
    hypothesis: str
    changes_from_previous: str
    expected_improvement: str
    model: str
    params: dict[str, Any]
    feature_recipe: str
    stop_criteria: str


# ── Learner registry (scikit-uplift model names + safe estimator defaults) ────

_LEARNER_FAMILIES = set(REGISTERED_UPLIFT_TEMPLATES.values()) | {
    "response_model", "solo_model", "two_model", "class_transformation",
}

_ESTIMATOR_DEFAULTS: dict[str, dict] = {
    "xgboost":            {"n_estimators": 100, "max_depth": 5,  "learning_rate": 0.1},
    "lightgbm":           {"n_estimators": 100, "max_depth": 5,  "learning_rate": 0.1, "verbose": -1},
    "catboost":           {"iterations":   100, "depth":     5,  "learning_rate": 0.1, "verbose": 0},
    "logistic_regression":{"C": 1.0, "max_iter": 1000},
}

_WARMUP_ORDER = ["response_model", "solo_model", "two_model", "class_transformation"]


# ── Agent 1: CaseRetrievalAgent ───────────────────────────────────────────────

class CaseRetrievalAgent:
    """Reads the experiment ledger and surfaces relevant prior evidence."""

    _SKILL = _load_skill("case_retrieval")

    def __init__(self, ledger: UpliftLedger, llm: ChatLLM) -> None:
        self._ledger = ledger
        self._llm    = llm

    def run(self) -> RetrievedContext:
        records: list[UpliftExperimentRecord] = self._ledger.load()

        if not records:
            return RetrievedContext(
                similar_recipes=[],
                supported_hypotheses=[],
                refuted_hypotheses=[],
                best_learner_family="response_model",
                failed_runs=[],
                summary="Cold start — no prior trial history found.",
            )

        summary_data = [
            {
                "run_id":           r.run_id,
                "learner_family":   r.uplift_learner_family,
                "base_estimator":   r.base_estimator,
                "feature_recipe_id": r.feature_recipe_id,
                "qini_auc":         r.qini_auc,
                "uplift_auc":       r.uplift_auc,
                "verdict":          r.verdict,
                "status":           r.status,
                "error":            r.error,
                "next_actions":     r.next_recommended_actions,
            }
            for r in records
        ]

        raw    = self._llm(self._SKILL, json.dumps(summary_data, indent=2))
        parsed = _parse_json(raw)

        return RetrievedContext(
            similar_recipes      = parsed.get("similar_recipes",      []),
            supported_hypotheses = parsed.get("supported_hypotheses", []),
            refuted_hypotheses   = parsed.get("refuted_hypotheses",   []),
            best_learner_family  = parsed.get("best_learner_family",  "response_model"),
            failed_runs          = parsed.get("failed_runs",          []),
            summary              = parsed.get("summary",              ""),
        )


# ── Agent 2: HypothesisReasoningAgent ────────────────────────────────────────

class HypothesisReasoningAgent:
    """Converts prior evidence into the next hypothesis action."""

    _SKILL = _load_skill("hypothesis_reasoning")

    def __init__(self, hypothesis_store: UpliftHypothesisStore, llm: ChatLLM) -> None:
        self._store = hypothesis_store
        self._llm   = llm

    def run(
        self,
        context: RetrievedContext,
        current_hypothesis: Optional[str] = None,
        latest_record: Optional[UpliftExperimentRecord] = None,
    ) -> HypothesisDecision:

        latest_summary = None
        if latest_record:
            latest_summary = {
                "qini_auc":   latest_record.qini_auc,
                "uplift_auc": latest_record.uplift_auc,
                "verdict":    latest_record.verdict,
                "next_actions": latest_record.next_recommended_actions,
            }

        user_msg = json.dumps({
            "retrieved_context": {
                "supported_hypotheses": context.supported_hypotheses,
                "refuted_hypotheses":   context.refuted_hypotheses,
                "best_learner_family":  context.best_learner_family,
                "summary":              context.summary,
            },
            "current_hypothesis":  current_hypothesis,
            "latest_trial_result": latest_summary,
        }, indent=2)

        raw    = self._llm(self._SKILL, user_msg)
        parsed = _parse_json(raw)

        decision = HypothesisDecision(
            action     = parsed.get("action",     "propose"),
            hypothesis = parsed.get("hypothesis", ""),
            evidence   = parsed.get("evidence",   ""),
            confidence = float(parsed.get("confidence", 0.5)),
        )

        # Update hypothesis store lifecycle
        self._sync_hypothesis_store(decision)
        return decision

    def _sync_hypothesis_store(self, decision: HypothesisDecision) -> None:
        if not decision.hypothesis:
            return
        existing = self._store.query_by_status("proposed") + self._store.query_by_status("under_test")
        match = next((h for h in existing if h.hypothesis_text == decision.hypothesis), None)

        if match is None and decision.action == "propose":
            new_h = UpliftHypothesis(
                question       = decision.hypothesis,
                hypothesis_text= decision.hypothesis,
                stage_origin   = "llm",
                action_type    = "recipe_comparison",
                expected_signal= decision.evidence,
                status         = "proposed",
            )
            self._store.append(new_h)


# ── Agent 3: UpliftStrategySelectionAgent ────────────────────────────────────

class UpliftStrategySelectionAgent:
    """Selects learner family, base estimator, feature recipe, and hyperparams."""

    _SKILL = _load_skill("uplift_strategy_selection")

    def __init__(self, ledger: UpliftLedger, llm: ChatLLM) -> None:
        self._ledger = ledger
        self._llm    = llm

    def run(self, hypothesis: HypothesisDecision, context: RetrievedContext) -> UpliftStrategy:
        records   = self._ledger.load()
        successful = [r for r in records if r.status == "success"]

        # Warm-up: one trial per learner family
        if len(successful) < len(_WARMUP_ORDER):
            used = {r.uplift_learner_family for r in successful}
            family = next((f for f in _WARMUP_ORDER if f not in used), _WARMUP_ORDER[0])
            return UpliftStrategy(
                learner_family = family,
                base_estimator = "logistic_regression",
                feature_recipe = "rfm_baseline",
                split_seed     = 42,
                eval_cutoff    = 0.3,
                rationale      = f"Warm-up trial for {family} with default params.",
            )

        # Optimization: LLM chooses from best-performing families
        family_auuc: dict[str, list[float]] = {}
        for r in successful:
            fam  = r.uplift_learner_family
            auuc = r.qini_auc or 0.0
            family_auuc.setdefault(fam, []).append(auuc)

        mean_auuc  = {f: sum(v) / len(v) for f, v in family_auuc.items()}
        best_family = max(mean_auuc, key=mean_auuc.get)

        user_msg = json.dumps({
            "mean_auuc_by_family":  mean_auuc,
            "best_family_so_far":   best_family,
            "context_summary":      context.summary,
            "active_hypothesis":    hypothesis.hypothesis,
            "refuted_hypotheses":   context.refuted_hypotheses,
        }, indent=2)

        raw    = self._llm(self._SKILL, user_msg)
        parsed = _parse_json(raw)

        family = parsed.get("learner_family", best_family)
        if family not in _LEARNER_FAMILIES:
            family = best_family

        estimator = parsed.get("base_estimator", "logistic_regression")
        if estimator not in _ESTIMATOR_DEFAULTS:
            estimator = "logistic_regression"

        return UpliftStrategy(
            learner_family = family,
            base_estimator = estimator,
            feature_recipe = parsed.get("feature_recipe", "rfm_baseline"),
            split_seed     = int(parsed.get("split_seed", 42)),
            eval_cutoff    = float(parsed.get("eval_cutoff", 0.3)),
            rationale      = parsed.get("rationale", ""),
        )

    def estimator_params(self, estimator: str) -> dict:
        return dict(_ESTIMATOR_DEFAULTS.get(estimator, {}))


# ── Agent 4: TrialSpecWriterAgent ─────────────────────────────────────────────

class TrialSpecWriterAgent:
    """Produces a fully resolved, structured trial plan."""

    _SKILL = _load_skill("trial_spec_writer")

    def __init__(self, ledger: UpliftLedger, llm: ChatLLM) -> None:
        self._ledger = ledger
        self._llm    = llm

    def run(
        self,
        hypothesis: HypothesisDecision,
        strategy: UpliftStrategy,
        estimator_params: dict[str, Any],
    ) -> PlanningTrialSpec:

        records    = self._ledger.load()
        best       = max(records, key=lambda r: r.qini_auc or 0.0) if records else None
        trial_id   = f"UT-{uuid.uuid4().hex[:6]}"

        user_msg = json.dumps({
            "trial_id":            trial_id,
            "active_hypothesis":   hypothesis.hypothesis,
            "hypothesis_action":   hypothesis.action,
            "evidence":            hypothesis.evidence,
            "strategy":            asdict(strategy),
            "estimator_params":    estimator_params,
            "last_best_trial":     {
                "qini_auc": best.qini_auc,
                "verdict":  best.verdict,
            } if best else None,
        }, indent=2)

        raw    = self._llm(self._SKILL, user_msg)
        parsed = _parse_json(raw)

        return PlanningTrialSpec(
            trial_id             = trial_id,
            hypothesis           = parsed.get("hypothesis",           hypothesis.hypothesis),
            changes_from_previous= parsed.get("changes_from_previous","N/A"),
            expected_improvement = parsed.get("expected_improvement", "N/A"),
            model                = parsed.get("model", f"{strategy.learner_family} + {strategy.base_estimator}"),
            params               = parsed.get("params", estimator_params),
            feature_recipe       = parsed.get("feature_recipe", strategy.feature_recipe),
            stop_criteria        = parsed.get("stop_criteria",
                                             "Stop if AUUC does not improve by >0.01 over 3 consecutive trials."),
        )


# ── Orchestrator ──────────────────────────────────────────────────────────────

class ExperimentPlanningPhase:
    """
    Runs all four planning agents in order and returns a PlanningTrialSpec
    ready for the feature engineering + execution phase.

    Usage
    -----
        from src.uplift.llm_client import make_chat_llm
        from src.uplift.ledger import UpliftLedger
        from src.uplift.hypotheses import UpliftHypothesisStore
        from src.uplift.planning_agents import ExperimentPlanningPhase

        llm     = make_chat_llm("ollama", model="qwen2.5-coder:7b")
        ledger  = UpliftLedger("artifacts/ledger.jsonl")
        h_store = UpliftHypothesisStore("artifacts/hypotheses.jsonl")
        planner = ExperimentPlanningPhase(ledger, h_store, llm)
        spec    = planner.run()
    """

    def __init__(
        self,
        ledger: UpliftLedger,
        hypothesis_store: UpliftHypothesisStore,
        llm: ChatLLM,
    ) -> None:
        self._case_retrieval  = CaseRetrievalAgent(ledger, llm)
        self._hypothesis      = HypothesisReasoningAgent(hypothesis_store, llm)
        self._strategy        = UpliftStrategySelectionAgent(ledger, llm)
        self._trial_spec      = TrialSpecWriterAgent(ledger, llm)

    def run(self, current_hypothesis: Optional[str] = None) -> PlanningTrialSpec:
        print("[Planning] 1/4 Case Retrieval...")
        context = self._case_retrieval.run()
        print(f"        -> {context.summary}")

        print("[Planning] 2/4 Hypothesis Reasoning...")
        ledger_records = self._case_retrieval._ledger.load()
        latest = max(ledger_records, key=lambda r: r.qini_auc or 0.0) if ledger_records else None
        hyp    = self._hypothesis.run(context, current_hypothesis, latest)
        print(f"        -> [{hyp.action.upper()}] {hyp.hypothesis}")

        print("[Planning] 3/4 Strategy Selection...")
        strategy = self._strategy.run(hyp, context)
        print(f"        -> {strategy.learner_family} + {strategy.base_estimator} | {strategy.feature_recipe}")

        print("[Planning] 4/4 Trial Spec...")
        params = self._strategy.estimator_params(strategy.base_estimator)
        spec   = self._trial_spec.run(hyp, strategy, params)
        print(f"        -> {spec.trial_id} | {spec.model}")
        print(f"          hypothesis : {spec.hypothesis}")
        print(f"          stop when  : {spec.stop_criteria}")

        return spec
