"""Deterministic tuning helpers for uplift trials."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from src.models.uplift import UpliftExperimentRecord, UpliftTrialSpec
from src.uplift.ledger import params_hash
from src.uplift.llm_client import ChatLLM
from src.uplift.metrics import normalized_qini_auc_score


_REGULARIZED_PARAM_SETS: dict[str, list[dict[str, object]]] = {
    "logistic_regression": [
        {"C": 0.3, "max_iter": 1000},
        {"C": 1.0, "max_iter": 1000},
        {"C": 3.0, "max_iter": 1000},
    ],
    "gradient_boosting": [
        {
            "n_estimators": 200,
            "learning_rate": 0.03,
            "max_depth": 2,
            "min_samples_leaf": 50,
            "subsample": 0.7,
        },
        {
            "n_estimators": 120,
            "learning_rate": 0.03,
            "max_depth": 2,
            "min_samples_leaf": 100,
            "subsample": 0.7,
        },
    ],
    "random_forest": [
        {
            "n_estimators": 200,
            "max_depth": 6,
            "min_samples_leaf": 50,
            "max_features": "sqrt",
            "n_jobs": -1,
        },
        {
            "n_estimators": 300,
            "max_depth": 4,
            "min_samples_leaf": 100,
            "max_features": "sqrt",
            "n_jobs": -1,
        },
    ],
    "xgboost": [
        {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 10,
            "reg_lambda": 10.0,
        },
        {
            "n_estimators": 400,
            "max_depth": 2,
            "learning_rate": 0.03,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "min_child_weight": 20,
            "reg_lambda": 10.0,
        },
    ],
    "lightgbm": [
        {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "num_leaves": 15,
            "min_child_samples": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 10.0,
        },
        {
            "n_estimators": 400,
            "max_depth": 3,
            "learning_rate": 0.03,
            "num_leaves": 15,
            "min_child_samples": 100,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "reg_lambda": 10.0,
        },
    ],
}


@dataclass(frozen=True)
class TuningCandidate:
    """Internal AutoLift candidate selected for deterministic refinement."""

    rank: int
    source_run_id: str
    source_hypothesis_id: str
    template_name: str
    learner_family: str
    base_estimator: str
    feature_recipe_id: str
    split_seed: int
    score: float
    qini_auc: float | None
    held_out_qini_auc: float | None
    uplift_auc: float | None
    held_out_uplift_auc: float | None
    existing_params_hash: str


@dataclass(frozen=True)
class CandidateSearchSpace:
    """Validated LLM-proposed tuning room for one selected candidate."""

    candidate: TuningCandidate
    rationale: str
    search_space: dict[str, list[Any]]
    warnings: list[str]
    trial_budget: int


@dataclass(frozen=True)
class AgenticTuningPlan:
    """Dry-run plan for an auditable, deterministic agentic tuning phase."""

    tuning_seed: int
    top_k: int
    budget_rule: str
    llm_rationale: str
    candidates: list[TuningCandidate]
    search_spaces: list[CandidateSearchSpace]
    trial_specs: list[UpliftTrialSpec]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tuning_seed": self.tuning_seed,
            "top_k": self.top_k,
            "budget_rule": self.budget_rule,
            "selection_score_definition": (
                "min(validation, held_out) - 0.25 * abs(held_out - validation), "
                "using normalized Qini artifacts when available and raw Qini "
                "ledger metrics otherwise"
            ),
            "llm_rationale": self.llm_rationale,
            "candidates": [asdict(candidate) for candidate in self.candidates],
            "search_spaces": [
                {
                    "candidate": asdict(search_space.candidate),
                    "rationale": search_space.rationale,
                    "search_space": search_space.search_space,
                    "warnings": search_space.warnings,
                    "trial_budget": search_space.trial_budget,
                }
                for search_space in self.search_spaces
            ],
            "trial_specs": [
                {
                    "spec_id": spec.spec_id,
                    "hypothesis_id": spec.hypothesis_id,
                    "template_name": spec.template_name,
                    "learner_family": spec.learner_family,
                    "base_estimator": spec.base_estimator,
                    "feature_recipe_id": spec.feature_recipe_id,
                    "params": spec.params,
                    "split_seed": spec.split_seed,
                }
                for spec in self.trial_specs
            ],
            "warnings": self.warnings,
        }


@dataclass(frozen=True)
class _ParamRule:
    kind: str
    min_value: float | int | None = None
    max_value: float | int | None = None
    choices: tuple[Any, ...] = ()


_PARAM_GUARDRAILS: dict[str, dict[str, _ParamRule]] = {
    "logistic_regression": {
        "C": _ParamRule("float", 0.01, 10.0),
        "max_iter": _ParamRule("int", 200, 2000),
    },
    "lightgbm": {
        "n_estimators": _ParamRule("int", 150, 700),
        "learning_rate": _ParamRule("float", 0.01, 0.10),
        "max_depth": _ParamRule("int", 2, 6),
        "num_leaves": _ParamRule("int", 7, 63),
        "min_child_samples": _ParamRule("int", 30, 300),
        "subsample": _ParamRule("float", 0.50, 1.0),
        "colsample_bytree": _ParamRule("float", 0.50, 1.0),
        "reg_lambda": _ParamRule("float", 0.0, 30.0),
    },
    "xgboost": {
        "n_estimators": _ParamRule("int", 150, 700),
        "learning_rate": _ParamRule("float", 0.01, 0.10),
        "max_depth": _ParamRule("int", 2, 6),
        "subsample": _ParamRule("float", 0.50, 1.0),
        "colsample_bytree": _ParamRule("float", 0.50, 1.0),
        "min_child_weight": _ParamRule("float", 1.0, 50.0),
        "reg_lambda": _ParamRule("float", 0.0, 30.0),
        "gamma": _ParamRule("float", 0.0, 5.0),
    },
    "gradient_boosting": {
        "n_estimators": _ParamRule("int", 80, 400),
        "learning_rate": _ParamRule("float", 0.01, 0.10),
        "max_depth": _ParamRule("int", 2, 5),
        "min_samples_leaf": _ParamRule("int", 20, 200),
        "subsample": _ParamRule("float", 0.50, 1.0),
    },
    "random_forest": {
        "n_estimators": _ParamRule("int", 150, 500),
        "max_depth": _ParamRule("int", 3, 10),
        "min_samples_leaf": _ParamRule("int", 20, 200),
        "max_features": _ParamRule("choice", choices=("sqrt", "log2")),
        "n_jobs": _ParamRule("choice", choices=(-1,)),
    },
}

_DEFAULT_SEARCH_SPACES: dict[str, dict[str, list[Any]]] = {
    "logistic_regression": {
        "C": [0.1, 0.3, 1.0, 3.0],
        "max_iter": [1000],
    },
    "lightgbm": {
        "n_estimators": [300, 400, 500],
        "learning_rate": [0.02, 0.03, 0.05],
        "max_depth": [2, 3, 4],
        "num_leaves": [7, 15, 31],
        "min_child_samples": [50, 100, 200],
        "subsample": [0.6, 0.7, 0.8],
        "colsample_bytree": [0.6, 0.7, 0.8],
        "reg_lambda": [5.0, 10.0, 20.0],
    },
    "xgboost": {
        "n_estimators": [300, 400, 500],
        "learning_rate": [0.02, 0.03, 0.05],
        "max_depth": [2, 3, 4],
        "subsample": [0.6, 0.7, 0.8],
        "colsample_bytree": [0.6, 0.7, 0.8],
        "min_child_weight": [5.0, 10.0, 20.0],
        "reg_lambda": [5.0, 10.0, 20.0],
    },
    "gradient_boosting": {
        "n_estimators": [120, 200, 300],
        "learning_rate": [0.02, 0.03, 0.05],
        "max_depth": [2, 3],
        "min_samples_leaf": [50, 100],
        "subsample": [0.6, 0.7, 0.8],
    },
    "random_forest": {
        "n_estimators": [200, 300, 400],
        "max_depth": [4, 6, 8],
        "min_samples_leaf": [50, 100, 150],
        "max_features": ["sqrt"],
        "n_jobs": [-1],
    },
}

_MAX_VALUES_PER_PARAM = 5
_MAX_ENUMERATED_PARAM_COMBOS = 10_000


def build_pre_run_tuning_specs(
    base_spec: UpliftTrialSpec,
    *,
    split_seeds: Sequence[int] = (42, 7, 99, 123),
    max_param_sets: int = 2,
) -> list[UpliftTrialSpec]:
    """Expand one planned trial into deterministic param/seed candidates."""
    param_sets = _REGULARIZED_PARAM_SETS.get(base_spec.base_estimator, [base_spec.params])
    param_sets = param_sets[: max(1, max_param_sets)]
    specs: list[UpliftTrialSpec] = []
    for param_index, params in enumerate(param_sets, start=1):
        for seed in _unique_seeds(split_seeds, base_spec.split_seed):
            specs.append(
                base_spec.model_copy(
                    update={
                        "spec_id": f"{base_spec.spec_id}__tune_p{param_index}_s{seed}",
                        "hypothesis_id": f"{base_spec.hypothesis_id}__tune_p{param_index}_s{seed}",
                        "params": dict(params),
                        "split_seed": int(seed),
                    }
                )
            )
    return specs


def select_top_tuning_candidates(
    records: Iterable[UpliftExperimentRecord],
    *,
    top_k: int = 2,
) -> list[TuningCandidate]:
    """Select the strongest internal AutoLift candidates for refinement.

    The ranking intentionally uses only AutoLift ledger evidence. External
    comparison baselines belong in reporting, not in the tuning loop.
    """
    best_by_template_recipe: dict[
        tuple[str, str],
        tuple[float, UpliftExperimentRecord],
    ] = {}
    for record in records:
        if not _is_tunable_record(record):
            continue
        score = _agentic_candidate_score(record)
        if not math.isfinite(score):
            continue
        key = (record.template_name, record.feature_recipe_id)
        if (
            key not in best_by_template_recipe
            or score > best_by_template_recipe[key][0]
        ):
            best_by_template_recipe[key] = (score, record)

    candidate_rows = list(best_by_template_recipe.values())

    selected: list[TuningCandidate] = []
    for rank, (score, record) in enumerate(
        sorted(
            candidate_rows,
            key=lambda item: (item[0], item[1].template_name, item[1].run_id),
            reverse=True,
        )[: max(1, top_k)],
        start=1,
    ):
        selected.append(
            TuningCandidate(
                rank=rank,
                source_run_id=record.run_id,
                source_hypothesis_id=record.hypothesis_id,
                template_name=record.template_name,
                learner_family=record.uplift_learner_family,
                base_estimator=record.base_estimator,
                feature_recipe_id=record.feature_recipe_id,
                split_seed=record.split_seed,
                score=score,
                qini_auc=record.qini_auc,
                held_out_qini_auc=record.held_out_qini_auc,
                uplift_auc=record.uplift_auc,
                held_out_uplift_auc=record.held_out_uplift_auc,
                existing_params_hash=record.params_hash,
            )
        )
    return selected


def validate_tuning_search_space(
    base_estimator: str,
    proposed_search_space: dict[str, Any],
) -> tuple[dict[str, list[Any]], list[str]]:
    """Keep only estimator-compatible, bounded discrete tuning choices."""
    guardrails = _PARAM_GUARDRAILS.get(base_estimator, {})
    warnings: list[str] = []
    validated: dict[str, list[Any]] = {}

    for param_name, raw_values in proposed_search_space.items():
        rule = guardrails.get(param_name)
        if rule is None:
            warnings.append(f"Rejected unsupported tuning param {param_name!r}.")
            continue
        values = raw_values if isinstance(raw_values, list) else [raw_values]
        accepted: list[Any] = []
        for value in values:
            coerced = _coerce_param_value(value, rule)
            if coerced is None:
                warnings.append(
                    f"Rejected out-of-range value {value!r} for {param_name}."
                )
                continue
            if coerced not in accepted:
                accepted.append(coerced)
        if accepted:
            sorted_values = _sort_param_values(accepted)
            if len(sorted_values) > _MAX_VALUES_PER_PARAM:
                sorted_values = sorted_values[:_MAX_VALUES_PER_PARAM]
                warnings.append(
                    f"Trimmed {param_name} to {_MAX_VALUES_PER_PARAM} values."
                )
            validated[param_name] = sorted_values

    return dict(sorted(validated.items())), warnings


def build_agentic_tuning_plan(
    records: Iterable[UpliftExperimentRecord],
    *,
    llm: ChatLLM,
    tuning_seed: int = 20260501,
    top_k: int = 2,
    budget_multiplier: int = 4,
    max_trials_per_candidate: int = 16,
) -> AgenticTuningPlan:
    """Build a deterministic dry-run tuning plan from LLM-proposed search rooms."""
    candidates = select_top_tuning_candidates(records, top_k=top_k)
    proposal, proposal_warnings = _llm_tuning_proposal(candidates, llm)
    proposal_by_template = _proposal_by_template(proposal)
    llm_rationale = str(proposal.get("rationale") or "No LLM rationale supplied.")

    search_spaces: list[CandidateSearchSpace] = []
    trial_specs: list[UpliftTrialSpec] = []
    warnings = list(proposal_warnings)

    for candidate in candidates:
        raw_proposal = proposal_by_template.get(candidate.template_name, {})
        raw_space = raw_proposal.get("search_space") or raw_proposal.get("params") or {}
        if not isinstance(raw_space, dict):
            raw_space = {}
        validated, validation_warnings = validate_tuning_search_space(
            candidate.base_estimator,
            raw_space,
        )
        if not validated:
            validated, fallback_warnings = validate_tuning_search_space(
                candidate.base_estimator,
                _DEFAULT_SEARCH_SPACES.get(candidate.base_estimator, {}),
            )
            validation_warnings.extend(fallback_warnings)
            validation_warnings.append(
                f"Used default search space for {candidate.template_name}."
            )
        trial_budget = _trial_budget(
            len(validated),
            budget_multiplier=budget_multiplier,
            max_trials_per_candidate=max_trials_per_candidate,
        )
        candidate_space = CandidateSearchSpace(
            candidate=candidate,
            rationale=str(raw_proposal.get("rationale") or llm_rationale),
            search_space=validated,
            warnings=validation_warnings,
            trial_budget=trial_budget,
        )
        search_spaces.append(candidate_space)
        warnings.extend(validation_warnings)
        trial_specs.extend(
            _sample_tuning_specs(
                candidate_space,
                tuning_seed=tuning_seed,
                existing_hashes={candidate.existing_params_hash},
            )
        )

    return AgenticTuningPlan(
        tuning_seed=int(tuning_seed),
        top_k=int(top_k),
        budget_rule=(
            f"min({max_trials_per_candidate}, "
            f"{budget_multiplier} * tunable_parameter_count) per candidate"
        ),
        llm_rationale=llm_rationale,
        candidates=candidates,
        search_spaces=search_spaces,
        trial_specs=trial_specs,
        warnings=warnings,
    )


def write_agentic_tuning_plan(path: str | Path, plan: AgenticTuningPlan) -> str:
    """Persist the dry-run tuning plan as an auditable JSON artifact."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(plan.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return str(output_path)


def select_stable_tuning_record(
    records: Iterable[UpliftExperimentRecord],
) -> UpliftExperimentRecord | None:
    """Choose the candidate with the best stability-adjusted normalized Qini."""
    successful = [record for record in records if record.status == "success"]
    if not successful:
        return None
    return max(successful, key=_stable_record_score)


def tuning_summary(records: Iterable[UpliftExperimentRecord]) -> list[dict[str, object]]:
    """Return a compact summary for audit artifacts and logs."""
    rows: list[dict[str, object]] = []
    for record in records:
        val = _normalized_qini_from_record(record, "uplift_scores")
        held = _normalized_qini_from_record(record, "held_out_predictions")
        rows.append(
            {
                "run_id": record.run_id,
                "hypothesis_id": record.hypothesis_id,
                "status": record.status,
                "learner_family": record.uplift_learner_family,
                "base_estimator": record.base_estimator,
                "split_seed": record.split_seed,
                "params_hash": record.params_hash,
                "val_normalized_qini": val,
                "held_out_normalized_qini": held,
                "stable_score": _stable_record_score(record),
                "error": record.error,
            }
        )
    return rows


def write_tuning_summary(
    path: str | Path,
    records: Iterable[UpliftExperimentRecord],
) -> str:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(tuning_summary(records), indent=2),
        encoding="utf-8",
    )
    return str(output_path)


def _is_tunable_record(record: UpliftExperimentRecord) -> bool:
    if record.status != "success":
        return False
    hypothesis_id = record.hypothesis_id or ""
    if hypothesis_id == "manual_baseline":
        return False
    if "__tune_" in hypothesis_id or hypothesis_id.startswith("agentic_tune__"):
        return False
    if record.base_estimator not in _PARAM_GUARDRAILS:
        return False
    if record.uplift_learner_family == "random":
        return False
    return True


def _agentic_candidate_score(record: UpliftExperimentRecord) -> float:
    val = _normalized_qini_from_record(record, "uplift_scores")
    held = _normalized_qini_from_record(record, "held_out_predictions")
    if val is not None and held is not None:
        return min(val, held) - 0.25 * abs(held - val)
    if val is not None:
        return val
    if record.qini_auc is not None and record.held_out_qini_auc is not None:
        validation = float(record.qini_auc)
        holdout = float(record.held_out_qini_auc)
        return min(validation, holdout) - 0.25 * abs(holdout - validation)
    if record.held_out_qini_auc is not None:
        return float(record.held_out_qini_auc)
    if record.qini_auc is not None:
        return float(record.qini_auc)
    return float("-inf")


def _coerce_param_value(value: Any, rule: _ParamRule) -> Any | None:
    if isinstance(value, bool):
        return None
    if rule.kind == "choice":
        return value if value in rule.choices else None
    if rule.kind == "int":
        if isinstance(value, float):
            if not value.is_integer():
                return None
            value = int(value)
        if not isinstance(value, int):
            return None
        if rule.min_value is not None and value < rule.min_value:
            return None
        if rule.max_value is not None and value > rule.max_value:
            return None
        return int(value)
    if rule.kind == "float":
        if not isinstance(value, (int, float)):
            return None
        numeric = float(value)
        if rule.min_value is not None and numeric < float(rule.min_value):
            return None
        if rule.max_value is not None and numeric > float(rule.max_value):
            return None
        return numeric
    return None


def _sort_param_values(values: list[Any]) -> list[Any]:
    if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
        return sorted(values)
    return sorted(values, key=lambda value: (type(value).__name__, repr(value)))


def _llm_tuning_proposal(
    candidates: list[TuningCandidate],
    llm: ChatLLM,
) -> tuple[dict[str, Any], list[str]]:
    if not candidates:
        return {}, ["No tunable successful AutoLift candidates were available."]

    system = (
        "You are an AutoLift tuning planner. Propose small, discrete tuning "
        "search spaces using only the supplied AutoLift ledger candidates. "
        "Do not use any external benchmark, report result, or private score. "
        "Return strict JSON with keys: rationale and search_spaces."
    )
    payload = {
        "selection_policy": (
            "Candidates are already selected from internal validation/holdout "
            "stability. Propose bounded search rooms only."
        ),
        "budget_rule": "min(16, 4 * tunable_parameter_count) per candidate",
        "candidates": [
            {
                "rank": candidate.rank,
                "run_id": candidate.source_run_id,
                "hypothesis_id": candidate.source_hypothesis_id,
                "template_name": candidate.template_name,
                "learner_family": candidate.learner_family,
                "base_estimator": candidate.base_estimator,
                "feature_recipe_id": candidate.feature_recipe_id,
                "score": candidate.score,
                "qini_auc": candidate.qini_auc,
                "held_out_qini_auc": candidate.held_out_qini_auc,
                "uplift_auc": candidate.uplift_auc,
                "held_out_uplift_auc": candidate.held_out_uplift_auc,
            }
            for candidate in candidates
        ],
        "allowed_parameters": {
            estimator: {
                param: {
                    "kind": rule.kind,
                    "min": rule.min_value,
                    "max": rule.max_value,
                    "choices": list(rule.choices),
                }
                for param, rule in rules.items()
            }
            for estimator, rules in _PARAM_GUARDRAILS.items()
        },
        "required_shape": {
            "rationale": "short global rationale",
            "search_spaces": [
                {
                    "template_name": "candidate template_name",
                    "rationale": "why these knobs",
                    "search_space": {"param_name": ["small discrete values"]},
                }
            ],
        },
    }
    try:
        raw = llm(system, json.dumps(payload, sort_keys=True))
    except Exception as exc:
        return {}, [f"LLM tuning proposal failed; using defaults: {exc}"]
    return _parse_json_object(raw)


def _parse_json_object(raw: str) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    try:
        parsed = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        match = re.search(r"\{.*\}", raw or "", flags=re.DOTALL)
        if not match:
            return {}, ["LLM tuning proposal was not valid JSON; using defaults."]
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}, ["LLM tuning proposal JSON could not be parsed; using defaults."]
        warnings.append("Extracted JSON object from surrounding LLM text.")
    if not isinstance(parsed, dict):
        return {}, ["LLM tuning proposal was not a JSON object; using defaults."]
    return parsed, warnings


def _proposal_by_template(proposal: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_spaces = proposal.get("search_spaces", [])
    if not isinstance(raw_spaces, list):
        return {}
    mapped: dict[str, dict[str, Any]] = {}
    for item in raw_spaces:
        if not isinstance(item, dict):
            continue
        template_name = item.get("template_name")
        if isinstance(template_name, str) and template_name:
            mapped[template_name] = item
    return mapped


def _trial_budget(
    tunable_parameter_count: int,
    *,
    budget_multiplier: int,
    max_trials_per_candidate: int,
) -> int:
    if tunable_parameter_count <= 0:
        return 0
    return min(
        max(1, int(max_trials_per_candidate)),
        max(1, int(budget_multiplier) * int(tunable_parameter_count)),
    )


def _sample_tuning_specs(
    candidate_space: CandidateSearchSpace,
    *,
    tuning_seed: int,
    existing_hashes: set[str],
) -> list[UpliftTrialSpec]:
    if candidate_space.trial_budget <= 0 or not candidate_space.search_space:
        return []
    candidate = candidate_space.candidate
    keys = sorted(candidate_space.search_space)
    values_by_key = [candidate_space.search_space[key] for key in keys]
    if any(not values for values in values_by_key):
        return []

    seed_material = (
        f"{int(tuning_seed)}:{candidate.template_name}:"
        f"{candidate.source_run_id}:{candidate.existing_params_hash}"
    )
    seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    index_tuples = _sample_param_index_tuples(
        values_by_key,
        budget=candidate_space.trial_budget,
        rng=rng,
    )

    specs: list[UpliftTrialSpec] = []
    seen_effective_params: set[str] = set()
    for indexes in index_tuples:
        if len(specs) >= candidate_space.trial_budget:
            break
        params = {
            key: values_by_key[position][index]
            for position, (key, index) in enumerate(zip(keys, indexes, strict=True))
        }
        if params_hash(params) in existing_hashes:
            continue
        effective_key = _effective_params_hash(candidate.base_estimator, params)
        if effective_key in seen_effective_params:
            continue
        seen_effective_params.add(effective_key)
        serial = len(specs) + 1
        specs.append(
            UpliftTrialSpec(
                spec_id=(
                    f"AT-{candidate.rank:02d}-{serial:02d}-"
                    f"{_slug(candidate.template_name)}"
                ),
                hypothesis_id=(
                    f"agentic_tune__{candidate.source_hypothesis_id}__p{serial:02d}"
                ),
                template_name=candidate.template_name,
                learner_family=candidate.learner_family,  # type: ignore[arg-type]
                base_estimator=candidate.base_estimator,
                feature_recipe_id=candidate.feature_recipe_id,
                params=params,
                split_seed=candidate.split_seed,
            )
        )
    return specs


def _sample_param_index_tuples(
    values_by_key: list[list[Any]],
    *,
    budget: int,
    rng: random.Random,
) -> list[tuple[int, ...]]:
    total_combinations = math.prod(len(values) for values in values_by_key)
    index_ranges = [range(len(values)) for values in values_by_key]
    if total_combinations <= _MAX_ENUMERATED_PARAM_COMBOS:
        all_indexes = list(itertools.product(*index_ranges))
        rng.shuffle(all_indexes)
        return all_indexes

    selected: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    max_attempts = max(100, budget * 25)
    attempts = 0
    while (
        len(selected) < budget
        and len(seen) < total_combinations
        and attempts < max_attempts
    ):
        attempts += 1
        indexes = tuple(rng.randrange(len(values)) for values in values_by_key)
        if indexes in seen:
            continue
        seen.add(indexes)
        selected.append(indexes)
    return selected


def _effective_params_hash(base_estimator: str, params: dict[str, Any]) -> str:
    effective = dict(params)
    if base_estimator == "lightgbm":
        max_depth = effective.get("max_depth")
        num_leaves = effective.get("num_leaves")
        if isinstance(max_depth, int) and max_depth > 0 and isinstance(num_leaves, int):
            effective["num_leaves"] = min(num_leaves, 2 ** max_depth)
    return params_hash(effective)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:48] or "candidate"


def _stable_record_score(record: UpliftExperimentRecord) -> float:
    val = _normalized_qini_from_record(record, "uplift_scores")
    held = _normalized_qini_from_record(record, "held_out_predictions")
    if val is not None and held is not None:
        return min(val, held) - 0.25 * abs(held - val)
    if val is not None:
        return val
    if record.qini_auc is not None:
        return float(record.qini_auc)
    return float("-inf")


def _normalized_qini_from_record(
    record: UpliftExperimentRecord,
    artifact_key: str,
) -> float | None:
    path = record.artifact_paths.get(artifact_key)
    if not path:
        return None
    try:
        scores = pd.read_csv(path)
        return normalized_qini_auc_score(
            scores["target"].to_numpy(),
            scores["treatment_flg"].to_numpy(),
            scores["uplift"].to_numpy(),
        )
    except Exception:
        return None


def _unique_seeds(split_seeds: Sequence[int], fallback_seed: int) -> list[int]:
    seeds = [int(seed) for seed in split_seeds] or [int(fallback_seed)]
    if int(fallback_seed) not in seeds:
        seeds.insert(0, int(fallback_seed))
    return list(dict.fromkeys(seeds))
