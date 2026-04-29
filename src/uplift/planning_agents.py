"""PR2-style experiment planning phase.

This module keeps the teammate-facing names from PR2 while using the current
repo ledger and hypothesis store as the source of truth.
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from typing import get_args

from src.models.uplift import (
    UpliftActionType,
    UpliftExperimentRecord,
    UpliftHypothesis,
)
from src.uplift.hypotheses import UpliftHypothesisStore, transition_hypothesis
from src.uplift.ledger import UpliftLedger
from src.uplift.llm_client import ChatLLM
from src.uplift.templates import REGISTERED_UPLIFT_TEMPLATES

_SKILLS_DIR = Path(__file__).parent / "skills"
_VALID_ACTION_TYPES: frozenset[str] = frozenset(get_args(UpliftActionType))

_ESTIMATOR_DEFAULTS: dict[str, dict[str, Any]] = {
    "logistic_regression": {"C": 1.0, "max_iter": 1000},
    "gradient_boosting": {
        "n_estimators": 50,
        "learning_rate": 0.05,
        "max_depth": 2,
    },
    "xgboost": {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1},
    "lightgbm": {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1},
    "catboost": {"iterations": 100, "depth": 5, "learning_rate": 0.1},
}
_WARMUP_ORDER = ["response_model", "solo_model", "two_model", "class_transformation"]


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
    action: str
    hypothesis: str
    evidence: str
    confidence: float
    experiment_action_type: str = "recipe_comparison"


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
    """Readable planning output using the PR2 naming convention."""

    trial_id: str
    hypothesis: str
    learner_family: str
    base_estimator: str
    feature_recipe: str
    params: dict[str, Any]
    split_seed: int
    eval_cutoff: float
    changes_from_previous: str
    expected_improvement: str
    model: str
    stop_criteria: str


def _load_skill(name: str) -> str:
    """Load a skill prompt strictly. Missing files are a configuration bug, not a soft fallback."""
    path = _SKILLS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(
            f"Required skill prompt is missing: {path}. "
            f"Add the file or remove the agent that requires it."
        )
    text = path.read_text(encoding="utf-8")
    match = re.search(r"## System Prompt\s*\n+```[^\n]*\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def _parse_json_strict(text: str) -> dict:
    """Strict JSON parse: returns a dict or raises ValueError. No silent fallback."""
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
    stripped = re.sub(r"\s*```$", "", stripped)
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if not match:
            raise ValueError(f"no JSON object found in LLM output: {exc}") from exc
        try:
            payload = json.loads(match.group())
        except json.JSONDecodeError as inner:
            raise ValueError(f"invalid JSON in LLM output: {inner}") from inner
    if not isinstance(payload, dict):
        raise ValueError("LLM JSON payload must be an object")
    return payload


def _call_llm_strict(
    llm: ChatLLM,
    system: str,
    user: str,
    *,
    max_retries: int = 1,
) -> dict:
    """Call ``llm`` and parse its output as strict JSON; retry once on parse failure.

    On exhausted retries, raise ``ValueError``. Callers that can degrade gracefully
    should wrap this in their own try/except. Callers that produce executable trial
    specs should let the exception propagate.
    """
    last_error: Exception | None = None
    for _ in range(max_retries + 1):
        try:
            return _parse_json_strict(llm(system, user))
        except ValueError as exc:
            last_error = exc
    raise ValueError(
        f"LLM did not return valid JSON after {max_retries + 1} attempts: {last_error}"
    ) from last_error


class CaseRetrievalAgent:
    """Read prior ledger records and summarize useful planning evidence."""

    _SKILL = _load_skill("case_retrieval")

    def __init__(self, ledger: UpliftLedger, llm: ChatLLM) -> None:
        self.ledger = ledger
        self.llm = llm

    def run(self) -> RetrievedContext:
        records = self.ledger.load()
        if not records:
            return RetrievedContext(
                similar_recipes=[],
                supported_hypotheses=[],
                refuted_hypotheses=[],
                best_learner_family="response_model",
                failed_runs=[],
                summary="Cold start. No prior uplift trials are available.",
            )

        user_msg = json.dumps([_record_summary(r) for r in records])
        try:
            parsed = _call_llm_strict(self.llm, self._SKILL, user_msg)
        except ValueError:
            # Case retrieval is informational; failure must not block planning.
            parsed = {}
        return RetrievedContext(
            similar_recipes=parsed.get("similar_recipes", []),
            supported_hypotheses=parsed.get("supported_hypotheses", []),
            refuted_hypotheses=parsed.get("refuted_hypotheses", []),
            best_learner_family=parsed.get("best_learner_family", "response_model"),
            failed_runs=parsed.get("failed_runs", []),
            summary=parsed.get("summary", ""),
        )


class HypothesisReasoningAgent:
    """Choose the next hypothesis action and keep the hypothesis store in sync."""

    _SKILL = _load_skill("hypothesis_reasoning")

    def __init__(self, hypothesis_store: UpliftHypothesisStore, llm: ChatLLM) -> None:
        self.store = hypothesis_store
        self.llm = llm

    def run(
        self,
        context: RetrievedContext,
        current_hypothesis: str | None = None,
        latest_record: UpliftExperimentRecord | None = None,
    ) -> HypothesisDecision:
        user_msg = json.dumps(
            {
                "retrieved_context": asdict(context),
                "current_hypothesis": current_hypothesis,
                "latest_trial_result": _record_summary(latest_record)
                if latest_record is not None
                else None,
            },
            sort_keys=True,
        )
        # Strict parsing: hypothesis reasoning drives downstream trial specs,
        # so silent JSON failure would produce a garbage trial. Retry once,
        # then propagate.
        parsed = _call_llm_strict(self.llm, self._SKILL, user_msg)
        raw_action_type = parsed.get("experiment_action_type", "recipe_comparison")
        experiment_action_type = (
            raw_action_type if raw_action_type in _VALID_ACTION_TYPES else "recipe_comparison"
        )
        decision = HypothesisDecision(
            action=parsed.get("action", "propose"),
            hypothesis=parsed.get("hypothesis", current_hypothesis or ""),
            evidence=parsed.get("evidence", ""),
            confidence=float(parsed.get("confidence", 0.5)),
            experiment_action_type=experiment_action_type,
        )
        self._sync_hypothesis_store(decision)
        return decision

    def _sync_hypothesis_store(self, decision: HypothesisDecision) -> None:
        if not decision.hypothesis:
            return
        active = (
            self.store.query_by_status("proposed")
            + self.store.query_by_status("under_test")
        )
        match = next(
            (item for item in active if item.hypothesis_text == decision.hypothesis),
            None,
        )
        if match is None:
            hypothesis = UpliftHypothesis(
                question=decision.hypothesis,
                hypothesis_text=decision.hypothesis,
                stage_origin="llm",
                action_type=decision.experiment_action_type,
                expected_signal=decision.evidence or "improved qini_auc",
                status="proposed",
            )
            self.store.append(hypothesis)
            return
        if decision.action == "validate":
            self.store.append(transition_hypothesis(match, "supported"))
        elif decision.action == "refute":
            self.store.append(transition_hypothesis(match, "contradicted"))


class UpliftStrategySelectionAgent:
    """Pick learner family, estimator, feature recipe, and split seed."""

    _SKILL = _load_skill("uplift_strategy_selection")

    def __init__(self, ledger: UpliftLedger, llm: ChatLLM) -> None:
        self.ledger = ledger
        self.llm = llm

    def run(self, hypothesis: HypothesisDecision, context: RetrievedContext) -> UpliftStrategy:
        records = self.ledger.load()
        successful = [record for record in records if record.status == "success"]
        if len(successful) < len(_WARMUP_ORDER):
            used = {record.uplift_learner_family for record in successful}
            family = next((item for item in _WARMUP_ORDER if item not in used), _WARMUP_ORDER[0])
            return UpliftStrategy(
                learner_family=family,
                base_estimator="logistic_regression",
                feature_recipe="rfm_baseline",
                split_seed=42,
                eval_cutoff=0.3,
                rationale=f"Warm-up trial for {family}.",
            )

        mean_qini = _mean_qini_by_family(successful)
        best_family = max(mean_qini, key=mean_qini.get)
        # Strict parsing: strategy selection is the contract for which template
        # the trial executes. A garbage default would silently pick the wrong runner.
        parsed = _call_llm_strict(
            self.llm,
            self._SKILL,
            json.dumps(
                {
                    "mean_qini_by_family": mean_qini,
                    "best_family_so_far": best_family,
                    "context_summary": context.summary,
                    "active_hypothesis": hypothesis.hypothesis,
                },
                sort_keys=True,
            ),
        )
        family = _safe_learner_family(parsed.get("learner_family", best_family), best_family)
        estimator = parsed.get("base_estimator", "logistic_regression")
        if estimator not in _ESTIMATOR_DEFAULTS:
            estimator = "logistic_regression"
        return UpliftStrategy(
            learner_family=family,
            base_estimator=estimator,
            feature_recipe=parsed.get("feature_recipe", "rfm_baseline"),
            split_seed=int(parsed.get("split_seed", 42)),
            eval_cutoff=float(parsed.get("eval_cutoff", 0.3)),
            rationale=parsed.get("rationale", ""),
        )

    def estimator_params(self, estimator: str) -> dict[str, Any]:
        return dict(_ESTIMATOR_DEFAULTS.get(estimator, _ESTIMATOR_DEFAULTS["logistic_regression"]))


class TrialSpecWriterAgent:
    """Produce one readable PR2 planning trial spec."""

    _SKILL = _load_skill("trial_spec_writer")

    def __init__(self, ledger: UpliftLedger, llm: ChatLLM) -> None:
        self.ledger = ledger
        self.llm = llm

    def run(
        self,
        hypothesis: HypothesisDecision,
        strategy: UpliftStrategy,
        estimator_params: dict[str, Any],
    ) -> PlanningTrialSpec:
        trial_id = f"UT-{uuid.uuid4().hex[:6]}"
        # Strict parsing: trial spec output feeds directly into run_uplift_trials.
        parsed = _call_llm_strict(
            self.llm,
            self._SKILL,
            json.dumps(
                {
                    "trial_id": trial_id,
                    "active_hypothesis": hypothesis.hypothesis,
                    "hypothesis_action": hypothesis.action,
                    "evidence": hypothesis.evidence,
                    "strategy": asdict(strategy),
                    "estimator_params": estimator_params,
                },
                sort_keys=True,
            ),
        )
        return PlanningTrialSpec(
            trial_id=trial_id,
            hypothesis=parsed.get("hypothesis", hypothesis.hypothesis),
            learner_family=strategy.learner_family,
            base_estimator=strategy.base_estimator,
            feature_recipe=parsed.get("feature_recipe", strategy.feature_recipe),
            params=parsed.get("params", estimator_params),
            split_seed=strategy.split_seed,
            eval_cutoff=strategy.eval_cutoff,
            changes_from_previous=parsed.get("changes_from_previous", "N/A"),
            expected_improvement=parsed.get("expected_improvement", "N/A"),
            model=parsed.get(
                "model",
                f"{strategy.learner_family} + {strategy.base_estimator}",
            ),
            stop_criteria=parsed.get(
                "stop_criteria",
                "Stop if Qini AUC does not improve after 3 trials.",
            ),
        )


class ExperimentPlanningPhase:
    """Run PR2 planning agents in order and return a PlanningTrialSpec."""

    def __init__(
        self,
        ledger: UpliftLedger,
        hypothesis_store: UpliftHypothesisStore,
        llm: ChatLLM,
    ) -> None:
        self.case_retrieval = CaseRetrievalAgent(ledger, llm)
        self.hypothesis_reasoning = HypothesisReasoningAgent(hypothesis_store, llm)
        self.strategy_selection = UpliftStrategySelectionAgent(ledger, llm)
        self.trial_spec_writer = TrialSpecWriterAgent(ledger, llm)

    def run(self, current_hypothesis: str | None = None) -> PlanningTrialSpec:
        context = self.case_retrieval.run()
        records = self.case_retrieval.ledger.load()
        latest = (
            max(records, key=lambda record: record.qini_auc or float("-inf"))
            if records
            else None
        )
        hypothesis = self.hypothesis_reasoning.run(context, current_hypothesis, latest)
        strategy = self.strategy_selection.run(hypothesis, context)
        params = self.strategy_selection.estimator_params(strategy.base_estimator)
        return self.trial_spec_writer.run(hypothesis, strategy, params)


def _record_summary(record: UpliftExperimentRecord | None) -> dict[str, Any] | None:
    if record is None:
        return None
    return {
        "run_id": record.run_id,
        "learner_family": record.uplift_learner_family,
        "base_estimator": record.base_estimator,
        "feature_recipe_id": record.feature_recipe_id,
        "qini_auc": record.qini_auc,
        "uplift_auc": record.uplift_auc,
        "status": record.status,
        "verdict": record.verdict,
        "error": record.error,
        "next_actions": record.next_recommended_actions,
    }


def _mean_qini_by_family(records: list[UpliftExperimentRecord]) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for record in records:
        if record.qini_auc is not None:
            values.setdefault(record.uplift_learner_family, []).append(record.qini_auc)
    return {
        family: sum(scores) / len(scores)
        for family, scores in values.items()
        if scores
    } or {"response_model": 0.0}


def _safe_learner_family(candidate: str, fallback: str) -> str:
    executable_families = set(REGISTERED_UPLIFT_TEMPLATES.values()) - {"random"}
    return candidate if candidate in executable_families else fallback
