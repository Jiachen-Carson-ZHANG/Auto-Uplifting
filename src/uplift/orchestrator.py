"""PR1-style orchestration names over the current PR2/core uplift flow."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.models.uplift import (
    UpliftExperimentRecord,
    UpliftFeatureArtifact,
    UpliftProjectContract,
    UpliftTrialSpec,
)
from src.uplift.evaluation_agents import run_evaluation_phase
from src.uplift.hypotheses import UpliftHypothesisStore, transition_hypothesis
from src.uplift.ledger import UpliftLedger
from src.uplift.loop import run_uplift_trials
from src.uplift.llm_client import ChatLLM
from src.uplift.metrics import normalized_qini_auc_score
from src.uplift.planning_agents import ExperimentPlanningPhase, PlanningTrialSpec
from src.uplift.tuning import (
    build_pre_run_tuning_specs,
    select_stable_tuning_record,
    write_tuning_summary,
)


@dataclass(frozen=True)
class RetryDecision:
    should_continue: bool
    reason: str
    suggested_next_action: str


@dataclass(frozen=True)
class AutoLiftRunResult:
    trial_records: list[UpliftExperimentRecord]
    evaluation_results: list[dict]
    retry_decision: RetryDecision
    report_path: str | None
    benchmark_record: UpliftExperimentRecord | None = None


class RetryControllerAgent:
    """PR1 naming for simple retry/stop heuristics over the uplift ledger."""

    def __init__(
        self,
        ledger: UpliftLedger,
        *,
        max_trials: int = 5,
        flat_window: int = 3,
        flat_qini_threshold: float = 0.005,
        session_start_count: int = 0,
    ) -> None:
        self.ledger = ledger
        self.max_trials = max_trials
        self.flat_window = flat_window
        self.flat_qini_threshold = flat_qini_threshold
        self.session_start_count = session_start_count

    def run(self) -> RetryDecision:
        records = [
            record
            for record in self.ledger.load()
            if record.hypothesis_id != "manual_baseline"
            and "__tune_" not in (record.hypothesis_id or "")
        ]
        successful = [record for record in records if record.status == "success"]
        session_successful = len(successful) - self.session_start_count
        if session_successful >= self.max_trials:
            return RetryDecision(
                should_continue=False,
                reason=f"Reached max_trials={self.max_trials}.",
                suggested_next_action="generate_report",
            )
        duplicate = _first_duplicate_config(successful)
        if duplicate is not None:
            return RetryDecision(
                should_continue=False,
                reason=f"Duplicate config already ran: {duplicate.params_hash}.",
                suggested_next_action="change_strategy",
            )
        recent = [
            record.qini_auc
            for record in successful[-self.flat_window :]
            if record.qini_auc is not None
        ]
        if len(recent) == self.flat_window and max(recent) - min(recent) <= self.flat_qini_threshold:
            return RetryDecision(
                should_continue=False,
                reason="Qini AUC has flattened across recent trials.",
                suggested_next_action="generate_report",
            )
        return RetryDecision(
            should_continue=True,
            reason="Budget remains and recent trials still have information value.",
            suggested_next_action="plan_next_trial",
        )


class ManualBenchmarkAgent:
    """PR1 naming for a fixed human-readable baseline run."""

    def __init__(
        self,
        contract: UpliftProjectContract,
        *,
        feature_artifact: UpliftFeatureArtifact,
        output_dir: str | Path,
    ) -> None:
        self.contract = contract
        self.feature_artifact = feature_artifact
        self.output_dir = Path(output_dir)

    def run(self) -> UpliftExperimentRecord:
        spec = UpliftTrialSpec(
            spec_id="manual_baseline",
            hypothesis_id="manual_baseline",
            template_name="two_model_sklearn",
            learner_family="two_model",
            base_estimator="logistic_regression",
            feature_recipe_id=self.feature_artifact.feature_recipe_id,
            params={},
        )
        result = run_uplift_trials(
            self.contract,
            feature_artifact=self.feature_artifact,
            trial_specs=[spec],
            output_dir=self.output_dir,
        )
        return result.records[0]


class ReportingAgent:
    """PR1 naming for the final report surface."""

    def __init__(
        self,
        contract: UpliftProjectContract,
        ledger: UpliftLedger,
        *,
        output_path: str | Path,
        retry_decision: RetryDecision | None = None,
        evaluation_results: list[dict] | None = None,
    ) -> None:
        self.contract = contract
        self.ledger = ledger
        self.output_path = Path(output_path)
        self.retry_decision = retry_decision
        self.evaluation_results = list(evaluation_results or [])

    def agent_champion(self) -> UpliftExperimentRecord | None:
        agent_records = [
            record
            for record in self.ledger.load()
            if record.status == "success" and record.hypothesis_id != "manual_baseline"
        ]
        if not agent_records:
            return None
        return max(
            agent_records,
            key=_report_champion_metric,
        )

    def manual_benchmark(self) -> UpliftExperimentRecord | None:
        for record in self.ledger.load():
            if record.hypothesis_id == "manual_baseline":
                return record
        return None

    def run(self) -> str:
        champion = self.agent_champion()
        benchmark = self.manual_benchmark()
        # Use the evaluation result that matches the champion, not just the latest trial.
        champion_eval = _champion_eval_result(champion, self.evaluation_results)
        latest_policy = _latest_policy_result([champion_eval] if champion_eval else self.evaluation_results)
        latest_xai = _latest_xai_result([champion_eval] if champion_eval else self.evaluation_results)
        latest_judge = _latest_judge_result([champion_eval] if champion_eval else self.evaluation_results)
        lines = [
            "# AutoLift Experiment Report",
            "",
            f"Task: {self.contract.task_name}",
            "",
            "## Decision",
            "",
            _decision_line(champion, benchmark),
        ]
        caution = _heldout_caution(champion, benchmark)
        if caution:
            lines.extend(["", f"- {caution}"])
        lines.extend(["", "## Agent Champion", ""])
        if champion is None:
            lines.append("No successful agent trial is available yet.")
        else:
            lines.extend(
                [
                    f"- Run ID: {champion.run_id}",
                    f"- Template: {champion.template_name}",
                    f"- Learner family: {champion.uplift_learner_family}",
                    f"- Base estimator: {champion.base_estimator}",
                    f"- Normalized Qini AUC: {_format_metric(_normalized_qini_from_record(champion))}",
                    f"- Uplift AUC: {champion.uplift_auc}",
                    f"- Held-out Normalized Qini AUC: {_format_metric(_normalized_qini_from_record(champion, held_out=True))}",
                    f"- Held-out Uplift AUC: {champion.held_out_uplift_auc}",
                    f"- Policy gain: {champion.policy_gain}",
                ]
            )
        lines.extend(["", "## Manual Benchmark", ""])
        if benchmark is None:
            lines.append("Manual benchmark has not been run.")
        else:
            lines.extend(
                [
                    f"- Run ID: {benchmark.run_id}",
                    f"- Template: {benchmark.template_name}",
                    f"- Normalized Qini AUC: {_format_metric(_normalized_qini_from_record(benchmark))}",
                    f"- Uplift AUC: {benchmark.uplift_auc}",
                    f"- Held-out Normalized Qini AUC: {_format_metric(_normalized_qini_from_record(benchmark, held_out=True))}",
                    f"- Held-out Uplift AUC: {benchmark.held_out_uplift_auc}",
                ]
            )
        lines.extend(["", "## All Trials", ""])
        lines.extend(_trial_table_lines(self.ledger.load()))
        lines.extend(["", "## Seed Stability", ""])
        lines.extend(_seed_stability_lines(self.ledger.load()))
        lines.extend(["", "## Feature Semantics", ""])
        lines.extend(_feature_semantics_lines(self.ledger.load()))
        lines.extend(["", "## Policy Recommendation", ""])
        if latest_policy is None:
            lines.append("No policy simulation result is available yet.")
        else:
            policy_data = latest_policy.get("policy_data", {})
            lines.extend(
                [
                    f"- Recommended threshold: {latest_policy.get('recommended_threshold')}%",
                    f"- Rationale: {latest_policy.get('recommendation_rationale', 'See policy simulation outputs.')}",
                    f"- Segment summary: {policy_data.get('segment_summary', {})}",
                    f"- First targeting cutoff: {_first_targeting_result(policy_data)}",
                ]
            )
        lines.extend(["", "## Explanation", ""])
        if latest_xai is None:
            lines.append("No XAI result is available yet.")
        elif latest_xai.get("skipped"):
            lines.append(f"XAI skipped: {latest_xai.get('reason', 'not available')}")
        else:
            top_features = latest_xai.get("global_top_features") or latest_xai.get("top_features") or []
            narrative = latest_xai.get("narrative", "")
            lines.extend(
                [
                    f"- Method: {latest_xai.get('method', 'cached_model_permutation' if top_features else 'ledger_narrative')}",
                    f"- Top drivers: {top_features if top_features else narrative}",
                    f"- Leakage flag: {latest_xai.get('leakage_auto_flag', False)}",
                ]
            )
        lines.extend(["", "## Representative Cases", ""])
        if latest_xai is None or latest_xai.get("skipped"):
            lines.append("Representative cases are not available yet.")
        else:
            lines.append(str(latest_xai.get("representative_cases", {})))
        lines.extend(["", "## Hypothesis Loop", ""])
        if latest_judge is None:
            lines.append("No judge result is available yet.")
        else:
            lines.extend(
                [
                    f"- Judge verdict: {latest_judge.get('verdict', 'inconclusive')}",
                    f"- Metric evidence: {latest_judge.get('computed_metrics', {})}",
                    f"- Key evidence: {latest_judge.get('key_evidence', [])}",
                ]
            )
        if latest_policy is not None:
            lines.append(
                f"- New hypothesis suggestion: {latest_policy.get('new_hypothesis')}"
            )
        lines.extend(["", "## Retry Decision", ""])
        if self.retry_decision is None:
            lines.append("Retry decision was not attached to this report.")
        else:
            lines.extend(
                [
                    f"- Continue: {self.retry_decision.should_continue}",
                    f"- Reason: {self.retry_decision.reason}",
                    f"- Next action: {self.retry_decision.suggested_next_action}",
                ]
            )
        lines.extend(
            [
                "",
                "## Why this answer is credible",
                "",
                "- Fixed reference benchmark is reported separately from agent champion selection.",
                "- Qini values in this report are normalized by the report-facing perfect-oracle Qini denominator.",
                "- Trial records come from the append-only uplift ledger.",
                "- Policy recommendations are derived from saved uplift_scores.csv artifacts.",
                "- XAI is used as supporting evidence for hypothesis explanation, not as the metric source of truth.",
                "",
                "## Trial Count",
                "",
                f"{len(self.ledger.load())} ledger records.",
            ]
        )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return str(self.output_path)


class AutoLiftOrchestrator:
    """PR1 end-to-end name using PR2 planning/evaluation and the tested kernel."""

    def __init__(
        self,
        *,
        contract: UpliftProjectContract,
        planner: ExperimentPlanningPhase,
        feature_artifacts_by_name: dict[str, UpliftFeatureArtifact],
        output_dir: str | Path,
        llm: ChatLLM,
        run_benchmark: bool = True,
        retry_max_trials: int = 5,
        enable_pre_run_tuning: bool = False,
        tuning_split_seeds: tuple[int, ...] = (42, 7, 99, 123),
        tuning_max_param_sets: int = 2,
    ) -> None:
        self.contract = contract
        self.planner = planner
        self.feature_artifacts_by_name = dict(feature_artifacts_by_name)
        self.output_dir = Path(output_dir)
        self.llm = llm
        self.run_benchmark = run_benchmark
        self.retry_max_trials = retry_max_trials
        self.enable_pre_run_tuning = enable_pre_run_tuning
        self.tuning_split_seeds = tuning_split_seeds
        self.tuning_max_param_sets = tuning_max_param_sets
        self.ledger = planner.case_retrieval.ledger

    def run(self, max_iterations: int = 1) -> AutoLiftRunResult:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Benchmark dedup: reuse existing record if already run in a prior session
        benchmark = None
        if self.run_benchmark:
            existing_benchmark = next(
                (r for r in self.ledger.load() if r.hypothesis_id == "manual_baseline"),
                None,
            )
            if existing_benchmark is not None:
                benchmark = existing_benchmark
                print(f"[benchmark] reusing existing record {benchmark.run_id}")
            else:
                benchmark = ManualBenchmarkAgent(
                    self.contract,
                    feature_artifact=self._default_feature_artifact(),
                    output_dir=self.output_dir,
                ).run()

        # Count pre-existing successful main (non-tuning, non-baseline) trials so the
        # retry controller only counts trials added in this session against max_trials.
        session_start_count = sum(
            1 for r in self.ledger.load()
            if r.hypothesis_id != "manual_baseline"
            and "__tune_" not in (r.hypothesis_id or "")
            and r.status == "success"
        )

        trial_records: list[UpliftExperimentRecord] = []
        evaluation_results: list[dict] = []
        retry = RetryDecision(True, "Not evaluated yet.", "plan_next_trial")
        for _ in range(max_iterations):
            planning_spec = self.planner.run()
            feature_artifact = self._feature_artifact(planning_spec.feature_recipe)
            trial_spec = _trial_from_planning_spec(planning_spec, feature_artifact)
            trial_spec = self._tune_trial_spec_if_enabled(trial_spec, feature_artifact)
            strategy_rationale = (
                f"{planning_spec.changes_from_previous} | Expected: {planning_spec.expected_improvement}"
                if planning_spec.changes_from_previous or planning_spec.expected_improvement
                else ""
            )
            print(
                f"[plan] {trial_spec.spec_id}"
                f" | family={trial_spec.learner_family}"
                f" | estimator={trial_spec.base_estimator}"
                f" | recipe={planning_spec.feature_recipe}"
                f" | recipe_id={trial_spec.feature_recipe_id}"
                f" | temporal_policy={planning_spec.temporal_policy or getattr(feature_artifact, 'temporal_policy', '')}"
                f" | hypothesis={planning_spec.hypothesis[:80] if planning_spec.hypothesis else 'n/a'}"
                f" | changes={planning_spec.changes_from_previous[:80]}"
            )
            loop_result = run_uplift_trials(
                self.contract,
                feature_artifact=feature_artifact,
                trial_specs=[trial_spec],
                output_dir=self.output_dir,
            )
            record = loop_result.records[0]
            trial_records.append(record)
            if record.status == "success":
                scores = pd.read_csv(record.artifact_paths["uplift_scores"])
                features = pd.read_csv(feature_artifact.artifact_path)
                eval_result = run_evaluation_phase(
                    trial_meta={
                        "spec_id": trial_spec.spec_id,
                        "learner_family": trial_spec.learner_family,
                        "hypothesis_text": planning_spec.hypothesis,
                        "feature_recipe": planning_spec.feature_recipe,
                        "feature_recipe_id": trial_spec.feature_recipe_id,
                        "temporal_policy": planning_spec.temporal_policy
                        or getattr(feature_artifact, "temporal_policy", ""),
                        "feature_semantics_rationale": planning_spec.feature_semantics_rationale,
                        "feature_expected_signal": planning_spec.feature_expected_signal,
                    },
                    scores_df=scores,
                    ledger=self.ledger,
                    llm=self.llm,
                    model_dir=Path(record.artifact_paths["model"]).parent
                    if "model" in record.artifact_paths
                    else None,
                    features_df=features,
                    trial_status=record.status,
                )
                evaluation_results.append(eval_result)
                # Write agent reasoning back into the ledger record
                judge = eval_result.get("judge", {})
                xai = eval_result.get("xai", {})
                policy = eval_result.get("policy", {})
                verdict = judge.get("verdict", "inconclusive")
                top_features = xai.get("global_top_features") or []
                xai_summary = (
                    ", ".join(f["feature"] for f in top_features[:5])
                    if top_features
                    else xai.get("narrative", "")
                )
                xai_diagnostic = xai.get("feature_semantics_diagnostic", {})
                xai_sanity_summary = (
                    f"age_dominance_warning={xai_diagnostic.get('age_dominance_warning')}; "
                    f"behavioral_top5_present={xai_diagnostic.get('behavioral_top5_present')}"
                    if xai_diagnostic
                    else ""
                )
                self.ledger.patch_record(
                    record.run_id,
                    verdict=verdict,
                    judge_narrative=judge.get("narrative") or judge.get("reasoning") or judge.get("rationale", ""),
                    xai_summary=xai_summary,
                    policy_narrative=policy.get("narrative") or policy.get("recommendation_rationale", ""),
                    strategy_rationale=strategy_rationale,
                    feature_semantics_rationale=planning_spec.feature_semantics_rationale,
                    feature_expected_signal=planning_spec.feature_expected_signal,
                    temporal_policy=planning_spec.temporal_policy
                    or getattr(feature_artifact, "temporal_policy", ""),
                    xai_sanity_summary=xai_sanity_summary,
                    next_recommended_actions=judge.get("next_recommended_actions", []),
                )
                _record_hypothesis_trial_result(
                    self.planner.hypothesis_reasoning.store,
                    planning_spec,
                    trial_spec.spec_id,
                    verdict,
                )
                # Log summary to pipeline output
                key_evidence = judge.get("key_evidence") or []
                print(
                    f"[eval] {trial_spec.spec_id}"
                    f" | verdict={verdict}"
                    f" | norm_qini={judge.get('computed_metrics', {}).get('normalized_qini_auc', 'n/a')}"
                    f" | xai_top={xai_summary[:60]}"
                    f" | evidence={key_evidence[:2]}"
                )
            retry = RetryControllerAgent(
                self.ledger,
                max_trials=self.retry_max_trials,
                session_start_count=session_start_count,
            ).run()
            if not retry.should_continue:
                break

        report_path = ReportingAgent(
            self.contract,
            self.ledger,
            output_path=self.output_dir / "final_report.md",
            retry_decision=retry,
            evaluation_results=evaluation_results,
        ).run()
        return AutoLiftRunResult(
            trial_records=trial_records,
            evaluation_results=evaluation_results,
            retry_decision=retry,
            report_path=report_path,
            benchmark_record=benchmark,
        )

    def _default_feature_artifact(self) -> UpliftFeatureArtifact:
        if "rfm_baseline" in self.feature_artifacts_by_name:
            return self.feature_artifacts_by_name["rfm_baseline"]
        return next(iter(self.feature_artifacts_by_name.values()))

    def _feature_artifact(self, feature_recipe: str) -> UpliftFeatureArtifact:
        if feature_recipe in self.feature_artifacts_by_name:
            return self.feature_artifacts_by_name[feature_recipe]
        for artifact in self.feature_artifacts_by_name.values():
            if artifact.feature_recipe_id == feature_recipe:
                return artifact
        available = ", ".join(sorted(self.feature_artifacts_by_name))
        raise ValueError(
            f"unknown feature_recipe: {feature_recipe!r}. "
            f"Available recipes: {available}"
        )

    def _tune_trial_spec_if_enabled(
        self,
        trial_spec: UpliftTrialSpec,
        feature_artifact: UpliftFeatureArtifact,
    ) -> UpliftTrialSpec:
        if not self.enable_pre_run_tuning:
            return trial_spec
        tuning_specs = build_pre_run_tuning_specs(
            trial_spec,
            split_seeds=self.tuning_split_seeds,
            max_param_sets=self.tuning_max_param_sets,
        )
        tuning_output = self.output_dir / "pre_run_tuning" / trial_spec.spec_id
        tuning_result = run_uplift_trials(
            self.contract,
            feature_artifact=feature_artifact,
            trial_specs=tuning_specs,
            output_dir=tuning_output,
        )
        write_tuning_summary(
            tuning_output / "tuning_summary.json",
            tuning_result.records,
        )
        selected = select_stable_tuning_record(tuning_result.records)
        if selected is None:
            return trial_spec
        selected_spec = next(
            (
                spec
                for spec in tuning_specs
                if spec.hypothesis_id == selected.hypothesis_id
            ),
            None,
        )
        return selected_spec or trial_spec


def _trial_from_planning_spec(
    planning_spec: PlanningTrialSpec,
    feature_artifact: UpliftFeatureArtifact,
) -> UpliftTrialSpec:
    return UpliftTrialSpec(
        spec_id=planning_spec.trial_id,
        hypothesis_id=planning_spec.trial_id,
        template_name=_template_name(
            planning_spec.learner_family,
            planning_spec.base_estimator,
        ),
        learner_family=planning_spec.learner_family,
        base_estimator=planning_spec.base_estimator,
        feature_recipe_id=feature_artifact.feature_recipe_id,
        params=_sanitize_planning_params(
            planning_spec.params,
            planning_spec.base_estimator,
        ),
        split_seed=planning_spec.split_seed,
    )


def _record_hypothesis_trial_result(
    store: UpliftHypothesisStore,
    planning_spec: PlanningTrialSpec,
    trial_id: str,
    verdict: str,
) -> None:
    """Link the executed trial to its source hypothesis without brittle text matching."""
    hypothesis = None
    if planning_spec.source_hypothesis_id:
        hypothesis = store.get_latest(planning_spec.source_hypothesis_id)
    if hypothesis is None:
        active = (
            store.query_by_status("proposed")
            + store.query_by_status("under_test")
            + store.query_by_status("inconclusive")
        )
        hypothesis = next(
            (item for item in active if item.hypothesis_text == planning_spec.hypothesis),
            None,
        )
    if hypothesis is None:
        return
    if hypothesis.status in {"supported", "contradicted", "retired"}:
        return

    if hypothesis.status == "proposed":
        hypothesis = store.append(
            transition_hypothesis(hypothesis, "under_test", trial_ids=[trial_id])
        )
    elif hypothesis.status == "inconclusive":
        hypothesis = store.append(
            transition_hypothesis(hypothesis, "under_test", trial_ids=[trial_id])
        )
    elif trial_id not in hypothesis.trial_ids:
        hypothesis = store.append(
            transition_hypothesis(hypothesis, hypothesis.status, trial_ids=[trial_id])
        )

    if verdict == "supported":
        store.append(transition_hypothesis(hypothesis, "supported", trial_ids=[trial_id]))
    elif verdict in {"contradicted", "refuted"}:
        store.append(transition_hypothesis(hypothesis, "contradicted", trial_ids=[trial_id]))
    elif verdict == "inconclusive" and hypothesis.status == "under_test":
        store.append(transition_hypothesis(hypothesis, "inconclusive", trial_ids=[trial_id]))


def _sanitize_planning_params(params: dict, base_estimator: str) -> dict:
    """Keep only estimator-compatible params from LLM planning output."""
    execution_owned = {"random_state", "random_seed", "seed"}
    allowed_by_estimator = {
        "logistic_regression": {"C", "max_iter", "solver", "penalty", "class_weight"},
        "gradient_boosting": {
            "n_estimators",
            "learning_rate",
            "max_depth",
            "min_samples_leaf",
            "subsample",
            "max_features",
        },
        "random_forest": {
            "n_estimators",
            "max_depth",
            "min_samples_leaf",
            "max_features",
            "class_weight",
            "bootstrap",
            "n_jobs",
        },
        "xgboost": {
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "reg_lambda",
            "reg_alpha",
            "min_child_weight",
            "gamma",
        },
        "lightgbm": {
            "n_estimators",
            "max_depth",
            "learning_rate",
            "num_leaves",
            "subsample",
            "colsample_bytree",
            "reg_lambda",
            "reg_alpha",
            "min_child_samples",
        },
        "catboost": {"iterations", "depth", "learning_rate", "l2_leaf_reg"},
    }
    allowed = allowed_by_estimator.get(base_estimator, set())
    return {
        key: value
        for key, value in params.items()
        if key not in execution_owned
        and (not allowed or key in allowed)
        and _is_valid_planning_param(base_estimator, key, value)
    }


def _is_valid_planning_param(base_estimator: str, key: str, value) -> bool:
    positive_int_keys = {
        "n_estimators",
        "max_depth",
        "min_samples_leaf",
        "max_iter",
        "num_leaves",
        "iterations",
        "depth",
        "min_child_samples",
    }
    positive_float_keys = {"C", "learning_rate", "l2_leaf_reg"}
    nonnegative_float_keys = {"reg_lambda", "reg_alpha", "gamma", "min_child_weight"}
    unit_float_keys = {"subsample", "colsample_bytree"}

    if key in positive_int_keys:
        return _is_int(value) and value > 0
    if key in positive_float_keys:
        return _is_number(value) and value > 0
    if key in nonnegative_float_keys:
        return _is_number(value) and value >= 0
    if key in unit_float_keys:
        return _is_number(value) and 0 < float(value) <= 1
    if key == "n_jobs":
        return _is_int(value) and (value == -1 or value > 0)
    if key == "bootstrap":
        return isinstance(value, bool)
    if key == "solver":
        return value in {"liblinear", "lbfgs", "newton-cg", "sag", "saga"}
    if key == "penalty":
        return value in {"l1", "l2", "elasticnet", "none", None}
    if key == "class_weight":
        return value in {"balanced", None} or isinstance(value, dict)
    if key == "max_features":
        return (
            value is None
            or value in {"sqrt", "log2"}
            or (_is_int(value) and value > 0)
            or (_is_number(value) and 0 < float(value) <= 1)
        )
    return base_estimator not in {
        "logistic_regression",
        "gradient_boosting",
        "random_forest",
        "xgboost",
        "lightgbm",
        "catboost",
    }


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _template_name(learner_family: str, base_estimator: str) -> str:
    if learner_family == "response_model" and base_estimator == "gradient_boosting":
        return "response_model_gradient_boosting_sklearn"
    if learner_family == "response_model":
        return "response_model_sklearn"
    if learner_family == "solo_model" and base_estimator == "gradient_boosting":
        return "solo_model_gradient_boosting_sklearn"
    if learner_family == "solo_model" and base_estimator == "random_forest":
        return "solo_model_random_forest_sklearn"
    if learner_family == "solo_model" and base_estimator == "xgboost":
        return "solo_model_xgboost"
    if learner_family == "solo_model" and base_estimator == "lightgbm":
        return "solo_model_lightgbm"
    if learner_family == "solo_model":
        return "solo_model_sklearn"
    if learner_family == "two_model" and base_estimator == "gradient_boosting":
        return "two_model_gradient_boosting_sklearn"
    if learner_family == "two_model" and base_estimator == "random_forest":
        return "two_model_random_forest_sklearn"
    if learner_family == "two_model" and base_estimator == "xgboost":
        return "two_model_xgboost"
    if learner_family == "two_model" and base_estimator == "lightgbm":
        return "two_model_lightgbm"
    if learner_family == "two_model" and base_estimator == "catboost":
        return "two_model_catboost"
    if learner_family == "two_model":
        return "two_model_sklearn"
    if learner_family == "class_transformation" and base_estimator == "gradient_boosting":
        return "class_transformation_gradient_boosting_sklearn"
    if learner_family == "class_transformation" and base_estimator == "random_forest":
        return "class_transformation_random_forest_sklearn"
    if learner_family == "class_transformation" and base_estimator == "xgboost":
        return "class_transformation_xgboost"
    if learner_family == "class_transformation" and base_estimator == "lightgbm":
        return "class_transformation_lightgbm"
    if learner_family == "class_transformation":
        return "class_transformation_sklearn"
    raise ValueError(f"unsupported learner_family for execution: {learner_family}")


def _first_duplicate_config(records: list[UpliftExperimentRecord]) -> UpliftExperimentRecord | None:
    seen: set[tuple[str, str, str, str]] = set()
    for record in records:
        key = (
            record.params_hash,
            record.uplift_learner_family,
            record.base_estimator,
            record.feature_recipe_id,
        )
        if key in seen:
            return record
        seen.add(key)
    return None


def _trial_table_lines(records: list[UpliftExperimentRecord]) -> list[str]:
    lines = [
        "| Run | Role | Learner | Estimator | Val Normalized Qini | Val Uplift AUC | Held-out Normalized Qini | Held-out Uplift AUC |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for record in records:
        role = "Manual" if record.hypothesis_id == "manual_baseline" else "Agent"
        val_qini = _normalized_qini_from_record(record)
        held_out_qini = _normalized_qini_from_record(record, held_out=True)
        lines.append(
            "| "
            f"{record.run_id} | "
            f"{role} | "
            f"{record.uplift_learner_family} | "
            f"{record.base_estimator} | "
            f"{_format_metric(val_qini)} | "
            f"{_format_metric(record.uplift_auc)} | "
            f"{_format_metric(held_out_qini)} | "
            f"{_format_metric(record.held_out_uplift_auc)} |"
        )
    return lines


def _feature_semantics_lines(records: list[UpliftExperimentRecord]) -> list[str]:
    successful = [
        record
        for record in records
        if record.status == "success" and record.hypothesis_id != "manual_baseline"
    ]
    if not successful:
        return ["No agent feature semantics have been evaluated yet."]

    by_recipe: dict[str, list[UpliftExperimentRecord]] = {}
    for record in successful:
        by_recipe.setdefault(record.feature_recipe_id, []).append(record)

    lines = [
        "| Feature Recipe ID | Temporal Policy | Best Validation Normalized Qini | Intended Signal | XAI Check |",
        "|---|---|---:|---|---|",
    ]
    for recipe_id, recipe_records in sorted(by_recipe.items()):
        best = max(recipe_records, key=_report_champion_metric)
        validation = _normalized_qini_from_record(best)
        intended = best.feature_expected_signal or "Not captured."
        xai_check = best.xai_sanity_summary or best.xai_summary or "Not captured."
        temporal = best.temporal_policy or "not captured"
        lines.append(
            "| "
            f"{recipe_id} | "
            f"{temporal} | "
            f"{_format_metric(validation)} | "
            f"{_truncate_cell(intended)} | "
            f"{_truncate_cell(xai_check)} |"
        )
    return lines


def _truncate_cell(value: str, limit: int = 120) -> str:
    value = " ".join(str(value).replace("|", "/").split())
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _report_champion_metric(record: UpliftExperimentRecord) -> float:
    normalized = _normalized_qini_from_record(record)
    if normalized is not None:
        return normalized
    return record.qini_auc if record.qini_auc is not None else float("-inf")


def _normalized_qini_from_record(
    record: UpliftExperimentRecord,
    *,
    held_out: bool = False,
) -> float | None:
    artifact_key = "held_out_predictions" if held_out else "uplift_scores"
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


def _format_metric(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def _seed_stability_lines(records: list[UpliftExperimentRecord]) -> list[str]:
    groups: dict[tuple[str, str, str, str], list[UpliftExperimentRecord]] = {}
    for record in records:
        if record.status != "success" or record.hypothesis_id == "manual_baseline":
            continue
        key = (
            record.uplift_learner_family,
            record.base_estimator,
            record.feature_recipe_id,
            record.params_hash,
        )
        groups.setdefault(key, []).append(record)

    rows: list[tuple[float, str]] = []
    for group_records in groups.values():
        if len(group_records) < 2:
            continue
        ordered = sorted(group_records, key=lambda record: record.split_seed)
        val_scores = [_normalized_qini_from_record(record) for record in ordered]
        held_scores = [
            _normalized_qini_from_record(record, held_out=True)
            for record in ordered
        ]
        val_scores = [score for score in val_scores if score is not None]
        held_scores = [score for score in held_scores if score is not None]
        if not val_scores and not held_scores:
            continue
        mean_val = _mean_or_none(val_scores)
        mean_held = _mean_or_none(held_scores)
        min_held = min(held_scores) if held_scores else None
        held_range = (
            max(held_scores) - min(held_scores)
            if len(held_scores) >= 2
            else None
        )
        verdict = _stability_verdict(min_held, held_range)
        seeds = ", ".join(str(record.split_seed) for record in ordered)
        first = ordered[0]
        sort_key = mean_held if mean_held is not None else float("-inf")
        rows.append(
            (
                sort_key,
                "| "
                f"{first.uplift_learner_family} | "
                f"{first.base_estimator} | "
                f"{seeds} | "
                f"{len(ordered)} | "
                f"{_format_metric(mean_val)} | "
                f"{_format_metric(mean_held)} | "
                f"{_format_metric(min_held)} | "
                f"{_format_metric(held_range)} | "
                f"{verdict} |",
            )
        )

    if not rows:
        return ["No repeated-seed stability groups are available yet."]
    lines = [
        "| Learner | Estimator | Seeds | Runs | Mean Val Normalized Qini | Mean Held-out Normalized Qini | Min Held-out Normalized Qini | Held-out Range | Verdict |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    lines.extend(row for _, row in sorted(rows, key=lambda item: item[0], reverse=True))
    return lines


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _stability_verdict(min_held: float | None, held_range: float | None) -> str:
    if min_held is None or held_range is None:
        return "needs more seeds"
    if min_held >= 0.20 and held_range <= 0.05:
        return "stable"
    if min_held >= 0.15 and held_range <= 0.10:
        return "watch"
    return "unstable"


def _heldout_caution(
    champion: UpliftExperimentRecord | None,
    benchmark: UpliftExperimentRecord | None,
) -> str:
    if (
        champion is None
        or benchmark is None
        or _normalized_qini_from_record(champion, held_out=True) is None
        or _normalized_qini_from_record(benchmark, held_out=True) is None
    ):
        return ""
    champion_qini = _normalized_qini_from_record(champion, held_out=True)
    benchmark_qini = _normalized_qini_from_record(benchmark, held_out=True)
    if champion_qini is None or benchmark_qini is None:
        return ""
    if benchmark_qini > champion_qini:
        return (
            "Manual benchmark has stronger held-out normalized Qini "
            f"({benchmark_qini:.6f} vs {champion_qini:.6f}); "
            "treat the validation champion as provisional until more unique agent "
            "trials confirm stability."
        )
    return ""


def _decision_line(
    champion: UpliftExperimentRecord | None,
    benchmark: UpliftExperimentRecord | None,
) -> str:
    if champion is None:
        return "No agent champion is available yet."
    if benchmark is None:
        return f"Use the current agent champion {champion.run_id} as the best available tested answer."
    champion_qini = _report_champion_metric(champion)
    benchmark_qini = _report_champion_metric(benchmark)
    if champion_qini == float("-inf") or benchmark_qini == float("-inf"):
        if benchmark.qini_auc is None or champion.qini_auc is None:
            return f"Use the current agent champion {champion.run_id} as the best available tested answer."
        champion_qini = champion.qini_auc
        benchmark_qini = benchmark.qini_auc
    delta = champion_qini - benchmark_qini
    if delta >= 0:
        return (
            f"Use agent champion {champion.run_id}; it beats the manual benchmark "
            f"by {delta:.4f} validation normalized Qini AUC."
        )
    return (
        f"Keep manual benchmark as the current safety reference; agent champion "
        f"{champion.run_id} trails by {abs(delta):.4f} validation normalized Qini AUC."
    )


def _champion_eval_result(
    champion: UpliftExperimentRecord | None,
    evaluation_results: list[dict],
) -> dict | None:
    """Return eval result matching the champion, or a synthetic one from ledger fields.

    In resumed multi-session runs the champion may have been evaluated in a prior
    session whose eval result is not in the current in-memory list. In that case,
    reconstruct a minimal eval result from the narrative fields stored in the ledger.
    """
    if champion is None:
        return None
    for result in evaluation_results:
        judge = result.get("judge", {})
        if judge.get("trial_id") == champion.hypothesis_id:
            return result
    # Champion not in this session's memory — build from stored ledger narratives.
    if champion.judge_narrative or champion.xai_summary or champion.policy_narrative:
        return {
            "judge": {
                "verdict": champion.verdict,
                "narrative": champion.judge_narrative,
                "computed_metrics": {},
                "trial_id": champion.hypothesis_id,
            },
            "xai": {
                "narrative": champion.xai_summary,
                "global_top_features": [],
                "trial_id": champion.hypothesis_id,
                "skipped": not bool(champion.xai_summary),
                "leakage_auto_flag": False,
            },
            "policy": {
                "recommendation_rationale": champion.policy_narrative,
                "trial_id": champion.hypothesis_id,
            },
        }
    return None


def _latest_policy_result(evaluation_results: list[dict]) -> dict | None:
    for result in reversed(evaluation_results):
        policy = result.get("policy")
        if isinstance(policy, dict):
            return policy
    return None


def _latest_judge_result(evaluation_results: list[dict]) -> dict | None:
    for result in reversed(evaluation_results):
        if "judge" in result:
            return result["judge"]
    return None


def _latest_xai_result(evaluation_results: list[dict]) -> dict | None:
    for result in reversed(evaluation_results):
        xai = result.get("xai")
        if isinstance(xai, dict):
            return xai
    return None


def _first_targeting_result(policy_data: dict) -> dict:
    targeting = policy_data.get("targeting_results") or []
    return targeting[0] if targeting else {}
