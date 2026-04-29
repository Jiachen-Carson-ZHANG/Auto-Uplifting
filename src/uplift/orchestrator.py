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
from src.uplift.ledger import UpliftLedger
from src.uplift.loop import run_uplift_trials
from src.uplift.llm_client import ChatLLM
from src.uplift.planning_agents import ExperimentPlanningPhase, PlanningTrialSpec


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
    ) -> None:
        self.ledger = ledger
        self.max_trials = max_trials
        self.flat_window = flat_window
        self.flat_qini_threshold = flat_qini_threshold

    def run(self) -> RetryDecision:
        records = [
            record
            for record in self.ledger.load()
            if record.hypothesis_id != "manual_baseline"
        ]
        successful = [record for record in records if record.status == "success"]
        if len(successful) >= self.max_trials:
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
            key=lambda record: record.qini_auc
            if record.qini_auc is not None
            else float("-inf"),
        )

    def manual_benchmark(self) -> UpliftExperimentRecord | None:
        for record in self.ledger.load():
            if record.hypothesis_id == "manual_baseline":
                return record
        return None

    def run(self) -> str:
        champion = self.agent_champion()
        benchmark = self.manual_benchmark()
        latest_policy = _latest_policy_result(self.evaluation_results)
        latest_xai = _latest_xai_result(self.evaluation_results)
        latest_judge = _latest_judge_result(self.evaluation_results)
        lines = [
            "# AutoLift Experiment Report",
            "",
            f"Task: {self.contract.task_name}",
            "",
            "## Decision",
            "",
            _decision_line(champion, benchmark),
            "",
            "## Agent Champion",
            "",
        ]
        if champion is None:
            lines.append("No successful agent trial is available yet.")
        else:
            lines.extend(
                [
                    f"- Run ID: {champion.run_id}",
                    f"- Template: {champion.template_name}",
                    f"- Learner family: {champion.uplift_learner_family}",
                    f"- Base estimator: {champion.base_estimator}",
                    f"- Qini AUC: {champion.qini_auc}",
                    f"- Uplift AUC: {champion.uplift_auc}",
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
                    f"- Qini AUC: {benchmark.qini_auc}",
                    f"- Uplift AUC: {benchmark.uplift_auc}",
                ]
            )
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
            lines.extend(
                [
                    f"- Method: {latest_xai.get('method', 'unknown')}",
                    f"- Top drivers: {latest_xai.get('global_top_features') or latest_xai.get('top_features', [])}",
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
                "- Manual baseline is reported separately from agent champion selection.",
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
    ) -> None:
        self.contract = contract
        self.planner = planner
        self.feature_artifacts_by_name = dict(feature_artifacts_by_name)
        self.output_dir = Path(output_dir)
        self.llm = llm
        self.run_benchmark = run_benchmark
        self.ledger = planner.case_retrieval.ledger

    def run(self, max_iterations: int = 1) -> AutoLiftRunResult:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        benchmark = None
        if self.run_benchmark:
            benchmark = ManualBenchmarkAgent(
                self.contract,
                feature_artifact=self._default_feature_artifact(),
                output_dir=self.output_dir,
            ).run()

        trial_records: list[UpliftExperimentRecord] = []
        evaluation_results: list[dict] = []
        retry = RetryDecision(True, "Not evaluated yet.", "plan_next_trial")
        for _ in range(max_iterations):
            planning_spec = self.planner.run()
            feature_artifact = self._feature_artifact(planning_spec.feature_recipe)
            trial_spec = _trial_from_planning_spec(planning_spec, feature_artifact)
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
                evaluation_results.append(
                    run_evaluation_phase(
                        trial_meta={
                            "spec_id": trial_spec.spec_id,
                            "learner_family": trial_spec.learner_family,
                            "hypothesis_text": planning_spec.hypothesis,
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
                )
            retry = RetryControllerAgent(self.ledger).run()
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
        try:
            return self.feature_artifacts_by_name[feature_recipe]
        except KeyError as exc:
            raise ValueError(f"unknown feature_recipe: {feature_recipe}") from exc


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
        params=planning_spec.params,
        split_seed=planning_spec.split_seed,
    )


def _template_name(learner_family: str, base_estimator: str) -> str:
    if learner_family == "response_model" and base_estimator == "gradient_boosting":
        return "response_model_gradient_boosting_sklearn"
    if learner_family == "response_model":
        return "response_model_sklearn"
    if learner_family == "solo_model" and base_estimator == "gradient_boosting":
        return "solo_model_gradient_boosting_sklearn"
    if learner_family == "solo_model":
        return "solo_model_sklearn"
    if learner_family == "two_model" and base_estimator == "gradient_boosting":
        return "two_model_gradient_boosting_sklearn"
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
    if learner_family == "class_transformation":
        return "class_transformation_sklearn"
    raise ValueError(f"unsupported learner_family for execution: {learner_family}")


def _first_duplicate_config(records: list[UpliftExperimentRecord]) -> UpliftExperimentRecord | None:
    seen: set[tuple[str, str, str]] = set()
    for record in records:
        key = (
            record.params_hash,
            record.uplift_learner_family,
            record.feature_recipe_id,
        )
        if key in seen:
            return record
        seen.add(key)
    return None


def _decision_line(
    champion: UpliftExperimentRecord | None,
    benchmark: UpliftExperimentRecord | None,
) -> str:
    if champion is None:
        return "No agent champion is available yet."
    if benchmark is None or benchmark.qini_auc is None or champion.qini_auc is None:
        return f"Use the current agent champion {champion.run_id} as the best available tested answer."
    delta = champion.qini_auc - benchmark.qini_auc
    if delta >= 0:
        return (
            f"Use agent champion {champion.run_id}; it beats the manual benchmark "
            f"by {delta:.4f} validation Qini AUC."
        )
    return (
        f"Keep manual benchmark as the current safety reference; agent champion "
        f"{champion.run_id} trails by {abs(delta):.4f} validation Qini AUC."
    )


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
