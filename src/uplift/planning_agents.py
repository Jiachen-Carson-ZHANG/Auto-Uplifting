"""PR2-style experiment planning phase.

This module keeps the teammate-facing names from PR2 while using the current
repo ledger and hypothesis store as the source of truth.
"""
from __future__ import annotations

import json
import importlib.util
import re
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from typing import get_args

from src.models.uplift import (
    UpliftActionType,
    UpliftExperimentRecord,
    UpliftFeatureSemanticsDecision,
    UpliftHypothesis,
)
from src.uplift.hypotheses import UpliftHypothesisStore, transition_hypothesis
from src.uplift.ledger import UpliftLedger
from src.uplift.llm_client import ChatLLM
from src.uplift.templates import (
    REGISTERED_UPLIFT_TEMPLATE_BASE_ESTIMATORS,
    REGISTERED_UPLIFT_TEMPLATES,
)

_SKILLS_DIR = Path(__file__).parent / "skills"
_VALID_ACTION_TYPES: frozenset[str] = frozenset(get_args(UpliftActionType))

_ESTIMATOR_DEFAULTS: dict[str, dict[str, Any]] = {
    "logistic_regression": {"C": 1.0, "max_iter": 1000},
    "gradient_boosting": {
        "n_estimators": 200,
        "learning_rate": 0.03,
        "max_depth": 2,
        "min_samples_leaf": 50,
        "subsample": 0.7,
    },
    "random_forest": {
        "n_estimators": 300,
        "max_depth": 4,
        "min_samples_leaf": 100,
        "n_jobs": -1,
    },
    "xgboost": {
        "n_estimators": 400,
        "max_depth": 2,
        "learning_rate": 0.03,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_lambda": 10.0,
        "min_child_weight": 20,
    },
    "lightgbm": {
        "n_estimators": 400,
        "max_depth": 3,
        "learning_rate": 0.03,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "num_leaves": 15,
        "min_child_samples": 100,
        "reg_lambda": 10.0,
    },
    "catboost": {"iterations": 100, "depth": 5, "learning_rate": 0.1},
}
_OPTIONAL_ESTIMATOR_IMPORTS = {
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "catboost": "catboost",
}
_MINIMAL_WARMUP_CANDIDATES: list[tuple[str, str]] = []
_POST_WARMUP_EXPLORATION_PRIORITY = [
    ("class_transformation", "logistic_regression"),
    ("class_transformation", "gradient_boosting"),
    ("class_transformation", "xgboost"),
    ("class_transformation", "lightgbm"),
    ("solo_model", "xgboost"),
    ("solo_model", "lightgbm"),
    ("two_model", "gradient_boosting"),
    ("two_model", "xgboost"),
    ("two_model", "lightgbm"),
    ("solo_model", "gradient_boosting"),
    ("two_model", "random_forest"),
    ("class_transformation", "random_forest"),
    ("solo_model", "random_forest"),
    ("solo_model", "logistic_regression"),
]


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
    hypothesis_id: str = ""


@dataclass
class UpliftStrategy:
    learner_family: str
    base_estimator: str
    feature_recipe: str
    split_seed: int
    eval_cutoff: float
    rationale: str
    feature_semantics_rationale: str = ""
    expected_feature_signal: str = ""
    temporal_policy: str = ""
    xai_sanity_checks: list[str] | None = None


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
    feature_semantics_rationale: str = ""
    feature_expected_signal: str = ""
    temporal_policy: str = ""
    xai_sanity_checks: list[str] | None = None
    source_hypothesis_id: str = ""


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


class FeatureSemanticsAgent:
    """Choose an approved semantic feature recipe before model strategy selection."""

    _SKILL = _load_skill("feature_semantics")

    def __init__(self, ledger: UpliftLedger, llm: ChatLLM) -> None:
        self.ledger = ledger
        self.llm = llm

    def run(
        self,
        *,
        context: RetrievedContext,
        available_feature_recipes: Sequence[str],
    ) -> UpliftFeatureSemanticsDecision:
        approved = list(available_feature_recipes) or ["rfm_baseline"]
        parsed = _call_llm_strict(
            self.llm,
            self._SKILL,
            json.dumps(
                {
                    "available_feature_recipes": approved,
                    "context_summary": context.summary,
                    "prior_records": [
                        _record_summary(record) for record in self.ledger.load()
                    ],
                    "selection_policy": (
                        "Choose a recipe that can test whether richer behavioral "
                        "semantics reduce suspicious age-only XAI dominance."
                    ),
                },
                sort_keys=True,
            ),
        )
        decision = UpliftFeatureSemanticsDecision.model_validate(parsed)
        if decision.feature_recipe in approved:
            return decision
        fallback = "rfm_baseline" if "rfm_baseline" in approved else approved[0]
        return decision.model_copy(
            update={
                "feature_recipe": fallback,
                "rationale": (
                    f"LLM proposed unavailable recipe {decision.feature_recipe}; "
                    f"fell back to approved recipe {fallback}. {decision.rationale}"
                ),
            }
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
                "constraints": {
                    "excluded_champion_families": ["response_model", "random"],
                    "unsupported_learners": ["causal_forest"],
                    "available_model_pairs": _available_strategy_pairs(),
                    "current_feature_recipe": {
                        "name": "rfm_baseline",
                        "feature_groups": ["demographic", "rfm", "basket", "points"],
                        "note": "RFM-style recency, frequency, and monetary aggregates already exist.",
                    },
                },
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
        hypothesis_text = _sanitize_hypothesis_text(
            parsed.get("hypothesis", current_hypothesis or "")
        )
        evidence = _sanitize_hypothesis_text(parsed.get("evidence", ""))
        decision = HypothesisDecision(
            action=parsed.get("action", "propose"),
            hypothesis=hypothesis_text,
            evidence=evidence,
            confidence=float(parsed.get("confidence", 0.5)),
            experiment_action_type=experiment_action_type,
        )
        stored = self._sync_hypothesis_store(decision)
        if stored is not None:
            decision.hypothesis_id = stored.hypothesis_id
        return decision

    def _sync_hypothesis_store(
        self,
        decision: HypothesisDecision,
    ) -> UpliftHypothesis | None:
        if not decision.hypothesis:
            return None
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
            return self.store.append(hypothesis)
        if decision.action in {"validate", "refute"} and match.status == "proposed":
            return self.store.append(transition_hypothesis(match, "under_test"))
        return match


class UpliftStrategySelectionAgent:
    """Pick learner family, estimator, feature recipe, and split seed."""

    _SKILL = _load_skill("uplift_strategy_selection")

    def __init__(self, ledger: UpliftLedger, llm: ChatLLM) -> None:
        self.ledger = ledger
        self.llm = llm

    def run(
        self,
        hypothesis: HypothesisDecision,
        context: RetrievedContext,
        feature_decision: UpliftFeatureSemanticsDecision | None = None,
        available_feature_recipes: Sequence[str] | None = None,
    ) -> UpliftStrategy:
        records = self.ledger.load()
        successful_records = [record for record in records if record.status == "success"]
        successful = [
            record
            for record in successful_records
            if record.hypothesis_id != "manual_baseline"
        ]
        approved_recipes = list(available_feature_recipes or ["rfm_baseline"])
        selected_recipe = (
            feature_decision.feature_recipe
            if feature_decision is not None
            else "rfm_baseline"
        )
        if selected_recipe not in approved_recipes:
            selected_recipe = "rfm_baseline" if "rfm_baseline" in approved_recipes else approved_recipes[0]
        warmup_candidates = _available_autonomous_warmup_candidates()
        if len(successful) < len(warmup_candidates):
            used = {
                (record.uplift_learner_family, record.base_estimator)
                for record in successful
            }
            family, estimator = next(
                (
                    candidate
                    for candidate in warmup_candidates
                    if candidate not in used
                ),
                warmup_candidates[0],
            )
            return UpliftStrategy(
                learner_family=family,
                base_estimator=estimator,
                feature_recipe=selected_recipe,
                split_seed=42,
                eval_cutoff=0.3,
                rationale=f"Warm-up trial for {family} with {estimator}.",
                feature_semantics_rationale=feature_decision.rationale
                if feature_decision is not None
                else "",
                expected_feature_signal=feature_decision.expected_signal
                if feature_decision is not None
                else "",
                temporal_policy=feature_decision.temporal_policy
                if feature_decision is not None
                else "",
                xai_sanity_checks=feature_decision.xai_sanity_checks
                if feature_decision is not None
                else [],
            )

        mean_qini = _mean_qini_by_family(successful)
        best_family = max(mean_qini, key=mean_qini.get)
        used_pairs = _used_strategy_pairs(successful_records)
        unused_pairs = _unused_strategy_pairs(successful_records)
        # Strict parsing: strategy selection is the contract for which template
        # the trial executes. A garbage default would silently pick the wrong runner.
        parsed = _call_llm_strict(
            self.llm,
            self._SKILL,
            json.dumps(
                {
                    "available_model_pairs": _available_strategy_pairs(),
                    "used_model_pairs": [
                        [family, estimator] for family, estimator in sorted(used_pairs)
                    ],
                    "unused_model_pairs": [
                        [family, estimator] for family, estimator in unused_pairs
                    ],
                    "mean_qini_by_family": mean_qini,
                    "best_family_so_far": best_family,
                    "context_summary": context.summary,
                    "active_hypothesis": hypothesis.hypothesis,
                    "feature_semantics": feature_decision.model_dump()
                    if feature_decision is not None
                    else None,
                    "available_feature_recipes": approved_recipes,
                    "selection_policy": (
                        "After the minimal warmup, choose the next uplift "
                        "learner from validation ledger evidence only. Final "
                        "generalization audit is outside the adaptive planning "
                        "loop. Do not choose a pair from used_model_pairs while "
                        "unused_model_pairs is non-empty."
                    ),
                },
                sort_keys=True,
            ),
        )
        family, estimator = _safe_strategy_pair(
            parsed.get("learner_family", best_family),
            parsed.get("base_estimator", "gradient_boosting"),
            fallback_family=best_family,
            fallback_estimator="gradient_boosting",
        )
        family, estimator, replacement_note = _replace_used_strategy_pair(
            family,
            estimator,
            successful_records,
        )
        rationale = parsed.get("rationale", "")
        if replacement_note:
            rationale = f"{replacement_note} {rationale}".strip()
        feature_recipe = parsed.get("feature_recipe", selected_recipe)
        if feature_recipe not in approved_recipes:
            feature_recipe = selected_recipe
        return UpliftStrategy(
            learner_family=family,
            base_estimator=estimator,
            feature_recipe=feature_recipe,
            split_seed=int(parsed.get("split_seed", 42)),
            eval_cutoff=float(parsed.get("eval_cutoff", 0.3)),
            rationale=rationale,
            feature_semantics_rationale=feature_decision.rationale
            if feature_decision is not None
            else "",
            expected_feature_signal=feature_decision.expected_signal
            if feature_decision is not None
            else "",
            temporal_policy=feature_decision.temporal_policy
            if feature_decision is not None
            else "",
            xai_sanity_checks=feature_decision.xai_sanity_checks
            if feature_decision is not None
            else [],
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
                    "feature_semantics": {
                        "rationale": strategy.feature_semantics_rationale,
                        "expected_signal": strategy.expected_feature_signal,
                        "temporal_policy": strategy.temporal_policy,
                        "xai_sanity_checks": strategy.xai_sanity_checks or [],
                    },
                },
                sort_keys=True,
            ),
        )
        changes_from_previous = parsed.get("changes_from_previous", "N/A")
        expected_improvement = parsed.get("expected_improvement", "N/A")
        if strategy.feature_semantics_rationale and strategy.feature_semantics_rationale not in changes_from_previous:
            changes_from_previous = (
                f"{changes_from_previous} Feature semantics: "
                f"{strategy.feature_semantics_rationale}"
            )
        if strategy.expected_feature_signal and strategy.expected_feature_signal not in expected_improvement:
            expected_improvement = (
                f"{expected_improvement} Expected feature signal: "
                f"{strategy.expected_feature_signal}"
            )
        return PlanningTrialSpec(
            trial_id=trial_id,
            hypothesis=parsed.get("hypothesis", hypothesis.hypothesis),
            learner_family=strategy.learner_family,
            base_estimator=strategy.base_estimator,
            feature_recipe=strategy.feature_recipe,
            params=parsed.get("params", estimator_params),
            split_seed=strategy.split_seed,
            eval_cutoff=strategy.eval_cutoff,
            changes_from_previous=changes_from_previous,
            expected_improvement=expected_improvement,
            model=parsed.get(
                "model",
                f"{strategy.learner_family} + {strategy.base_estimator}",
            ),
            stop_criteria=parsed.get(
                "stop_criteria",
                "Stop if Qini AUC does not improve after 3 trials.",
            ),
            feature_semantics_rationale=strategy.feature_semantics_rationale,
            feature_expected_signal=strategy.expected_feature_signal,
            temporal_policy=strategy.temporal_policy,
            xai_sanity_checks=strategy.xai_sanity_checks or [],
            source_hypothesis_id=hypothesis.hypothesis_id,
        )


class ExperimentPlanningPhase:
    """Run PR2 planning agents in order and return a PlanningTrialSpec."""

    def __init__(
        self,
        ledger: UpliftLedger,
        hypothesis_store: UpliftHypothesisStore,
        llm: ChatLLM,
        available_feature_recipes: Sequence[str] | None = None,
    ) -> None:
        self.case_retrieval = CaseRetrievalAgent(ledger, llm)
        self.feature_semantics = FeatureSemanticsAgent(ledger, llm)
        self.hypothesis_reasoning = HypothesisReasoningAgent(hypothesis_store, llm)
        self.strategy_selection = UpliftStrategySelectionAgent(ledger, llm)
        self.trial_spec_writer = TrialSpecWriterAgent(ledger, llm)
        self.available_feature_recipes = list(available_feature_recipes or ["rfm_baseline"])

    def run(self, current_hypothesis: str | None = None) -> PlanningTrialSpec:
        context = self.case_retrieval.run()
        records = self.case_retrieval.ledger.load()
        latest = (
            max(records, key=lambda record: record.qini_auc or float("-inf"))
            if records
            else None
        )
        feature_decision = self.feature_semantics.run(
            context=context,
            available_feature_recipes=self.available_feature_recipes,
        )
        hypothesis = self.hypothesis_reasoning.run(context, current_hypothesis, latest)
        strategy = self.strategy_selection.run(
            hypothesis,
            context,
            feature_decision,
            self.available_feature_recipes,
        )
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
        "xai_summary": record.xai_summary,
        "strategy_rationale": record.strategy_rationale,
        "feature_semantics_rationale": record.feature_semantics_rationale,
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
    } or {"two_model": 0.0}


def _sanitize_hypothesis_text(text: str) -> str:
    sanitized = str(text or "").strip()
    lower = sanitized.lower()
    mentions_response_only = (
        "response_model" in lower
        or "response model" in lower
        or "response-only" in lower
    )
    asserts_missing_rfm = "rfm" in lower and any(
        phrase in lower
        for phrase in [
            "no explicit",
            "absent",
            "lacks",
            "lack ",
            "without",
            "not include",
            "none",
        ]
    )
    mentions_unsupported_causal_forest = (
        "causal forest" in lower
        or "causal_forest" in lower
        or "causal-forest" in lower
    )
    if mentions_unsupported_causal_forest:
        return (
            "Compare registered random-forest and boosted uplift templates "
            "rather than unimplemented causal-forest learners."
        )
    if mentions_response_only or asserts_missing_rfm:
        return (
            "Compare eligible uplift learners on the existing RFM baseline "
            "feature recipe, prioritizing two-model and class-transformation "
            "variants with stronger tree-based estimators."
        )
    return sanitized


def _used_strategy_pairs(records: list[UpliftExperimentRecord]) -> set[tuple[str, str]]:
    return {
        (record.uplift_learner_family, record.base_estimator)
        for record in records
        if record.status == "success"
    }


def _unused_strategy_pairs(records: list[UpliftExperimentRecord]) -> list[tuple[str, str]]:
    used = _used_strategy_pairs(records)
    available = [tuple(pair) for pair in _available_strategy_pairs()]
    return [pair for pair in _rank_strategy_pairs(available) if pair not in used]


def _rank_strategy_pairs(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    priority = {
        pair: index for index, pair in enumerate(_POST_WARMUP_EXPLORATION_PRIORITY)
    }
    family_rank = {"two_model": 0, "class_transformation": 1, "solo_model": 2}
    estimator_rank = {
        "xgboost": 0,
        "lightgbm": 1,
        "random_forest": 2,
        "gradient_boosting": 3,
        "logistic_regression": 4,
    }
    return sorted(
        pairs,
        key=lambda pair: (
            priority.get(pair, 999),
            family_rank.get(pair[0], 999),
            estimator_rank.get(pair[1], 999),
            pair,
        ),
    )


def _replace_used_strategy_pair(
    family: str,
    estimator: str,
    records: list[UpliftExperimentRecord],
) -> tuple[str, str, str]:
    if (family, estimator) not in _used_strategy_pairs(records):
        return family, estimator, ""
    unused = _unused_strategy_pairs(records)
    if not unused:
        return (
            family,
            estimator,
            "All available model pairs already ran; allowing parameter variation.",
        )
    replacement_family, replacement_estimator = unused[0]
    return (
        replacement_family,
        replacement_estimator,
        (
            f"LLM proposed {family}+{estimator}, but that model pair already ran; "
            f"switched to {replacement_family}+{replacement_estimator}."
        ),
    )


def _safe_learner_family(candidate: str, fallback: str) -> str:
    executable_families = set(REGISTERED_UPLIFT_TEMPLATES.values()) - {
        "random",
        "response_model",
    }
    safe_fallback = fallback if fallback in executable_families else "two_model"
    return candidate if candidate in executable_families else safe_fallback


def _safe_base_estimator(candidate: str, fallback: str) -> str:
    safe_fallback = (
        fallback
        if fallback in _ESTIMATOR_DEFAULTS and _is_estimator_available(fallback)
        else "gradient_boosting"
    )
    if candidate not in _ESTIMATOR_DEFAULTS:
        return safe_fallback
    if not _is_estimator_available(candidate):
        return safe_fallback
    return candidate


def _safe_strategy_pair(
    candidate_family: str,
    candidate_estimator: str,
    *,
    fallback_family: str,
    fallback_estimator: str,
) -> tuple[str, str]:
    family = _safe_learner_family(candidate_family, fallback_family)
    estimator = _safe_base_estimator(candidate_estimator, fallback_estimator)
    available = {tuple(pair) for pair in _available_strategy_pairs()}
    if (family, estimator) in available:
        return family, estimator
    fallback_pair = (
        _safe_learner_family(fallback_family, "two_model"),
        _safe_base_estimator(fallback_estimator, "gradient_boosting"),
    )
    if fallback_pair in available:
        return fallback_pair
    return "two_model", "gradient_boosting"


def _available_autonomous_warmup_candidates() -> list[tuple[str, str]]:
    return [
        candidate
        for candidate in _MINIMAL_WARMUP_CANDIDATES
        if candidate in {tuple(pair) for pair in _available_strategy_pairs()}
    ]


def _available_strategy_pairs() -> list[list[str]]:
    pairs: list[tuple[str, str]] = []
    for template_name, family in REGISTERED_UPLIFT_TEMPLATES.items():
        if family in {"random", "response_model"}:
            continue
        estimator = REGISTERED_UPLIFT_TEMPLATE_BASE_ESTIMATORS.get(template_name)
        if estimator is None or not _is_estimator_available(estimator):
            continue
        pair = (family, estimator)
        if pair not in pairs:
            pairs.append(pair)
    return [[family, estimator] for family, estimator in sorted(pairs)]


def _is_estimator_available(estimator: str) -> bool:
    module_name = _OPTIONAL_ESTIMATOR_IMPORTS.get(estimator)
    return module_name is None or importlib.util.find_spec(module_name) is not None
