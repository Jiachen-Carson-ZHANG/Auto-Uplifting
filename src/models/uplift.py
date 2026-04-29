"""Uplift-specific contracts for campaign targeting experiments.

These models intentionally compose with generic V2 contracts rather than
subclassing them. Uplift semantics require treatment, outcome, scoring, and
policy fields that generic classification contracts should not absorb.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

UpliftActionType = Literal[
    "recipe_comparison",
    "window_sweep",
    "feature_ablation",
    "response_overlap_disambiguation",
    "ranking_stability_check",
    "cost_sensitivity",
    "feature_group_expansion",
]

UpliftHypothesisStatus = Literal[
    "proposed",
    "under_test",
    "supported",
    "contradicted",
    "inconclusive",
    "retired",
]

UpliftHypothesisOrigin = Literal[
    "eda",
    "diagnosis",
    "evaluation",
    "policy",
    "failure_analysis",
    "manual",
    "llm",
]

UpliftWaveCreatedBy = Literal["manual", "llm"]

UpliftWaveStatus = Literal["completed", "partially_completed", "blocked", "failed"]

UpliftStopReason = Literal[
    "validity_blocked",
    "compute_exhausted",
    "no_valid_next_action",
    "low_information_gain",
    "champion_stable",
    "policy_threshold_stable",
    "business_decision_supportable",
]

M2B_SUPPORTED_WAVE_ACTIONS = {
    "cost_sensitivity",
    "recipe_comparison",
    "ranking_stability_check",
    "window_sweep",
    "feature_ablation",
    "feature_group_expansion",
}


def _stable_hash(payload: object, length: int = 12) -> str:
    """Return a short deterministic SHA-256 hash for a JSON-serializable payload."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]


class UpliftTableSchema(BaseModel):
    """Paths for the canonical X5/RetailHero uplift tables."""

    clients_table: str
    purchases_table: str
    train_table: str
    scoring_table: str
    products_table: Optional[str] = None

    client_id_column: str = "client_id"
    treatment_column: str = "treatment_flg"
    target_column: str = "target"
    uplift_column: str = "uplift"


class UpliftSplitContract(BaseModel):
    """Internal labeled-data split policy for uplift evaluation."""

    train_fraction: float = 0.7
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    random_seed: int = 42
    stratification_policy: Literal[
        "joint_treatment_outcome",
        "treatment_only",
        "random",
    ] = "joint_treatment_outcome"
    min_rows_per_partition: int = 2

    @model_validator(mode="after")
    def _validate_fractions(self) -> "UpliftSplitContract":
        total = self.train_fraction + self.val_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-9:
            raise ValueError("train/val/test fractions must sum to 1.0")
        if min(self.train_fraction, self.val_fraction, self.test_fraction) < 0:
            raise ValueError("split fractions must be non-negative")
        if self.min_rows_per_partition < 1:
            raise ValueError("min_rows_per_partition must be >= 1")
        return self

    @property
    def n_requested_partitions(self) -> int:
        """Number of non-empty partitions requested by the split policy."""
        return sum(
            frac > 0
            for frac in [self.train_fraction, self.val_fraction, self.test_fraction]
        )

    @property
    def min_stratum_size(self) -> int:
        """Minimum total rows needed per stratum to populate all partitions."""
        return self.min_rows_per_partition * self.n_requested_partitions


class UpliftEvaluationPolicy(BaseModel):
    """Metric and business-value policy for uplift experiments."""

    primary_metric: str = "qini_auc"
    secondary_metrics: List[str] = Field(
        default_factory=lambda: ["uplift_auc", "uplift_at_k"]
    )
    higher_is_better: bool = True
    cutoff_grid: List[float] = Field(default_factory=lambda: [0.05, 0.10, 0.20, 0.30])

    conversion_value: Optional[float] = None
    communication_cost: Optional[float] = None
    budget_capacity: Optional[float] = None
    cost_scenarios: Dict[str, float] = Field(
        default_factory=lambda: {
            "zero_cost": 0.0,
            "low_cost": 0.05,
            "medium_cost": 0.20,
        }
    )

    @model_validator(mode="after")
    def _validate_cutoffs(self) -> "UpliftEvaluationPolicy":
        if not self.cutoff_grid:
            raise ValueError("cutoff_grid must not be empty")
        if any(c <= 0 or c > 1 for c in self.cutoff_grid):
            raise ValueError("cutoff_grid values must be in (0, 1]")
        return self


class UpliftFeatureRecipeSpec(BaseModel):
    """Deterministic description of a reusable uplift feature recipe."""

    source_tables: List[str]
    feature_groups: List[str]
    windows_days: List[int] = Field(default_factory=list)
    interactions: List[str] = Field(default_factory=list)
    builder_version: str = "v1"
    reference_date: Optional[str] = None
    artifact_path: Optional[str] = None

    @model_validator(mode="after")
    def _canonicalize_lists(self) -> "UpliftFeatureRecipeSpec":
        self.source_tables = sorted(dict.fromkeys(self.source_tables))
        self.feature_groups = sorted(dict.fromkeys(self.feature_groups))
        self.windows_days = sorted(dict.fromkeys(self.windows_days))
        self.interactions = sorted(dict.fromkeys(self.interactions))
        if self.reference_date is not None:
            self.reference_date = datetime.fromisoformat(self.reference_date).isoformat()
        return self

    def canonical_payload(self) -> Dict[str, object]:
        """Return the stable payload used for recipe hashing."""
        return {
            "source_tables": self.source_tables,
            "feature_groups": self.feature_groups,
            "windows_days": self.windows_days,
            "interactions": self.interactions,
            "builder_version": self.builder_version,
            "reference_date": self.reference_date,
        }

    @property
    def feature_recipe_id(self) -> str:
        """Stable ID derived from canonical recipe configuration."""
        return _stable_hash(self.canonical_payload())

    def compute_feature_artifact_id(self, dataset_fingerprint: str) -> str:
        """Stable ID for one materialized artifact from this recipe and dataset."""
        return _stable_hash(
            {
                "feature_recipe_id": self.feature_recipe_id,
                "dataset_fingerprint": dataset_fingerprint,
                "builder_version": self.builder_version,
            }
        )


class UpliftFeatureArtifact(BaseModel):
    """Metadata for one materialized customer-level uplift feature table."""

    feature_recipe_id: str
    feature_artifact_id: str
    dataset_fingerprint: str
    builder_version: str
    artifact_path: str
    metadata_path: str
    cohort: Literal["train", "scoring", "all"] = "train"
    entity_key: str = "client_id"
    reference_date: Optional[str] = None
    row_count: int
    columns: List[str]
    generated_columns: List[str]
    source_tables: List[str]
    feature_groups: List[str] = Field(default_factory=list)
    windows_days: List[int] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)

    @model_validator(mode="after")
    def _validate_artifact_metadata(self) -> "UpliftFeatureArtifact":
        if self.row_count < 0:
            raise ValueError("row_count must be non-negative")
        if not self.columns:
            raise ValueError("columns must not be empty")
        if self.entity_key not in self.columns:
            raise ValueError(f"feature artifact must include {self.entity_key}")
        self.source_tables = sorted(dict.fromkeys(self.source_tables))
        self.feature_groups = sorted(dict.fromkeys(self.feature_groups))
        self.windows_days = sorted(dict.fromkeys(self.windows_days))
        if self.reference_date is not None:
            self.reference_date = datetime.fromisoformat(self.reference_date).isoformat()
        return self


class UpliftHypothesis(BaseModel):
    """Pointer-only hypothesis record for Uplift Supervisor."""

    hypothesis_id: str = Field(default_factory=lambda: f"UH-{uuid4().hex[:8]}")
    question: str
    hypothesis_text: str
    stage_origin: UpliftHypothesisOrigin
    action_type: UpliftActionType
    expected_signal: str = ""
    status: UpliftHypothesisStatus = "proposed"
    wave_ids: List[str] = Field(default_factory=list)
    trial_ids: List[str] = Field(default_factory=list)
    next_action: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    @model_validator(mode="after")
    def _validate_hypothesis(self) -> "UpliftHypothesis":
        if not self.question.strip():
            raise ValueError("question must not be empty")
        if not self.hypothesis_text.strip():
            raise ValueError("hypothesis_text must not be empty")
        self.wave_ids = list(dict.fromkeys(self.wave_ids))
        self.trial_ids = list(dict.fromkeys(self.trial_ids))
        return self


class UpliftTrialSpec(BaseModel):
    """One executable uplift trial configuration."""

    spec_id: str = Field(default_factory=lambda: f"UT-{uuid4().hex[:6]}")
    hypothesis_id: str
    template_name: str
    learner_family: Literal[
        "random",
        "response_model",
        "two_model",
        "solo_model",
        "class_transformation",
    ]
    base_estimator: str = "logistic_regression"
    feature_recipe_id: str
    params: Dict[str, object] = Field(default_factory=dict)
    split_seed: int = 42
    primary_metric: str = "qini_auc"


class UpliftExperimentWaveSpec(BaseModel):
    """One supervisor wave composed of comparable uplift trial specs."""

    wave_id: str
    hypothesis_id: str
    action_type: UpliftActionType
    rationale: str
    trial_specs: List[UpliftTrialSpec]
    expected_signal: str
    success_criterion: str
    abort_on_first_failure: bool
    required_feature_recipe_ids: List[str]
    created_by: UpliftWaveCreatedBy

    @model_validator(mode="after")
    def _validate_wave_spec(self) -> "UpliftExperimentWaveSpec":
        if self.action_type not in M2B_SUPPORTED_WAVE_ACTIONS:
            raise ValueError(
                "supported wave actions are cost_sensitivity, recipe_comparison, "
                "ranking_stability_check, window_sweep, feature_ablation, "
                "and feature_group_expansion"
            )
        if not self.wave_id.strip():
            raise ValueError("wave_id must not be empty")
        if not self.hypothesis_id.strip():
            raise ValueError("hypothesis_id must not be empty")
        if not self.rationale.strip():
            raise ValueError("rationale must not be empty")
        if not self.expected_signal.strip():
            raise ValueError("expected_signal must not be empty")
        if not self.success_criterion.strip():
            raise ValueError("success_criterion must not be empty")

        self.required_feature_recipe_ids = list(
            dict.fromkeys(self.required_feature_recipe_ids)
        )
        if not self.required_feature_recipe_ids:
            raise ValueError("required_feature_recipe_ids must not be empty")

        if self.action_type in {"feature_ablation", "feature_group_expansion"}:
            if len(self.trial_specs) != 2:
                raise ValueError(
                    f"{self.action_type} waves require exactly 2 trial_specs"
                )
        elif not 2 <= len(self.trial_specs) <= 4:
            raise ValueError(
                f"{self.action_type} waves require 2 to 4 trial_specs"
            )

        spec_ids = [spec.spec_id for spec in self.trial_specs]
        if len(set(spec_ids)) != len(spec_ids):
            raise ValueError("trial_specs must have unique spec_id values")

        if any(spec.hypothesis_id != self.hypothesis_id for spec in self.trial_specs):
            raise ValueError("all trial_specs must use the same hypothesis_id")

        trial_recipe_ids = [spec.feature_recipe_id for spec in self.trial_specs]
        required_recipe_ids = set(self.required_feature_recipe_ids)
        if set(trial_recipe_ids) != required_recipe_ids:
            raise ValueError(
                "required_feature_recipe_ids must match trial feature_recipe_id values"
            )
        if self.action_type == "ranking_stability_check":
            if len(required_recipe_ids) != 1:
                raise ValueError(
                    "ranking_stability_check requires exactly one feature_recipe_id"
                )
            split_seeds = [spec.split_seed for spec in self.trial_specs]
            if len(set(split_seeds)) != len(split_seeds):
                raise ValueError(
                    "ranking_stability_check requires distinct split_seed values"
                )
        else:
            if len(set(trial_recipe_ids)) != len(trial_recipe_ids):
                raise ValueError(
                    f"{self.action_type} requires at least two distinct "
                    "feature_recipe_id values and no duplicate trial "
                    "feature_recipe_id values"
                )
            if len(set(trial_recipe_ids)) < 2:
                raise ValueError(
                    f"{self.action_type} requires at least two distinct "
                    "feature_recipe_id values"
                )

        first = self.trial_specs[0]
        comparable_fields = [
            "template_name",
            "learner_family",
            "base_estimator",
            "primary_metric",
        ]
        if self.action_type != "ranking_stability_check":
            comparable_fields.append("split_seed")
        for field_name in comparable_fields:
            expected = getattr(first, field_name)
            if any(getattr(spec, field_name) != expected for spec in self.trial_specs):
                raise ValueError(
                    f"{self.action_type} trials must use the same {field_name}"
                )
        return self


class UpliftWaveResult(BaseModel):
    """Pointer-only supervisor result for one completed or blocked wave."""

    wave_id: str
    hypothesis_id: str
    action_type: UpliftActionType
    status: UpliftWaveStatus
    trial_ids: List[str] = Field(default_factory=list)
    failed_trial_ids: List[str] = Field(default_factory=list)
    blocked_reason: Optional[str] = None
    champion_run_id: Optional[str] = None
    artifact_paths: Dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_wave_result(self) -> "UpliftWaveResult":
        if self.action_type not in M2B_SUPPORTED_WAVE_ACTIONS:
            raise ValueError(
                "supported wave actions are cost_sensitivity, recipe_comparison, "
                "ranking_stability_check, window_sweep, feature_ablation, "
                "and feature_group_expansion"
            )

        self.trial_ids = list(dict.fromkeys(self.trial_ids))
        self.failed_trial_ids = list(dict.fromkeys(self.failed_trial_ids))

        trial_id_set = set(self.trial_ids)
        failed_id_set = set(self.failed_trial_ids)
        if not failed_id_set.issubset(trial_id_set):
            raise ValueError("failed_trial_ids must be a subset of trial_ids")
        if self.champion_run_id is not None and self.champion_run_id not in trial_id_set:
            raise ValueError("champion_run_id must point to a trial run_id")

        if self.status == "completed":
            if not self.trial_ids:
                raise ValueError("completed waves require trial_ids")
            if self.failed_trial_ids:
                raise ValueError("completed waves require failed_trial_ids to be empty")
            if self.blocked_reason:
                raise ValueError("completed waves must not include blocked_reason")
            if self.champion_run_id is None:
                raise ValueError("completed waves require champion_run_id")
        elif self.status == "partially_completed":
            if not self.failed_trial_ids:
                raise ValueError(
                    "partially_completed waves require failed_trial_ids"
                )
            if self.champion_run_id is None:
                raise ValueError(
                    "partially_completed waves require champion_run_id"
                )
        elif self.status == "blocked":
            if not self.blocked_reason:
                raise ValueError("blocked waves require blocked_reason")
        elif self.status == "failed":
            if not self.failed_trial_ids and not self.blocked_reason:
                raise ValueError(
                    "failed waves require failed_trial_ids or blocked_reason"
                )
        return self


class UpliftStopDecision(BaseModel):
    """Pointer-only deterministic verdict for one uplift wave."""

    wave_id: str
    hypothesis_id: str
    action_type: UpliftActionType
    stop_reason: UpliftStopReason
    hypothesis_status: UpliftHypothesisStatus
    should_stop: bool
    trial_ids: List[str] = Field(default_factory=list)
    champion_run_id: Optional[str] = None
    next_action: Optional[str] = None
    evidence_summary: Dict[str, object] = Field(default_factory=dict)
    artifact_paths: Dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_stop_decision(self) -> "UpliftStopDecision":
        self.trial_ids = list(dict.fromkeys(self.trial_ids))
        if self.champion_run_id is not None and self.champion_run_id not in self.trial_ids:
            raise ValueError("champion_run_id must point to a trial run_id")
        if self.hypothesis_status == "proposed":
            raise ValueError("stop decisions must resolve beyond proposed")
        if self.stop_reason == "business_decision_supportable":
            if self.hypothesis_status != "supported":
                raise ValueError("business_decision_supportable requires supported")
            if not self.should_stop:
                raise ValueError("business_decision_supportable requires should_stop")
        return self


class UpliftRankingStabilityReport(BaseModel):
    """Rank agreement diagnostics across repeated champion-like runs."""

    run_ids: List[str]
    pair_count: int = 0
    min_rank_correlation: Optional[float] = None
    mean_rank_correlation: Optional[float] = None
    min_top_k_overlap: Optional[float] = None
    mean_top_k_overlap: Optional[float] = None
    stable: bool
    limitations: List[str] = Field(default_factory=list)


class UpliftPolicyStabilityReport(BaseModel):
    """Policy cutoff stability across configured cost scenarios."""

    run_ids: List[str]
    stable_cutoff_by_scenario: Dict[str, Optional[str]] = Field(default_factory=dict)
    unstable_scenarios: List[str] = Field(default_factory=list)
    stable: bool
    limitations: List[str] = Field(default_factory=list)


class UpliftRobustnessReport(BaseModel):
    """Combined M6 robustness evidence used by the deterministic stop policy."""

    run_ids: List[str]
    ranking: UpliftRankingStabilityReport
    policy: UpliftPolicyStabilityReport
    stable: bool
    limitations: List[str] = Field(default_factory=list)


class UpliftCandidateHypothesis(BaseModel):
    """Strict advisory hypothesis candidate proposed by diagnosis calls."""

    model_config = ConfigDict(extra="forbid")

    question: str
    hypothesis_text: str
    action_type: UpliftActionType
    expected_signal: str
    rationale: str

    @model_validator(mode="after")
    def _validate_candidate(self) -> "UpliftCandidateHypothesis":
        if not self.question.strip():
            raise ValueError("question must not be empty")
        if not self.hypothesis_text.strip():
            raise ValueError("hypothesis_text must not be empty")
        if not self.expected_signal.strip():
            raise ValueError("expected_signal must not be empty")
        if not self.rationale.strip():
            raise ValueError("rationale must not be empty")
        return self


class UpliftDiagnosisResult(BaseModel):
    """Strict advisory diagnosis output; it cannot mutate deterministic contracts."""

    model_config = ConfigDict(extra="forbid")

    unresolved_questions: List[str]
    risks: List[str] = Field(default_factory=list)
    candidate_hypotheses: List[UpliftCandidateHypothesis]

    @model_validator(mode="after")
    def _validate_diagnosis(self) -> "UpliftDiagnosisResult":
        self.unresolved_questions = [
            question.strip()
            for question in self.unresolved_questions
            if question.strip()
        ]
        self.risks = [risk.strip() for risk in self.risks if risk.strip()]
        if not self.unresolved_questions:
            raise ValueError("unresolved_questions must not be empty")
        if not self.candidate_hypotheses:
            raise ValueError("candidate_hypotheses must not be empty")
        return self


class UpliftAdvisoryVerdict(BaseModel):
    """LLM narrative around a deterministic stop decision."""

    model_config = ConfigDict(extra="forbid")

    stop_reason: UpliftStopReason
    hypothesis_status: UpliftHypothesisStatus
    verdict_summary: str
    rationale: str
    next_action: Optional[str] = None
    cited_artifact_paths: List[str]

    @model_validator(mode="after")
    def _validate_verdict(self) -> "UpliftAdvisoryVerdict":
        if not self.verdict_summary.strip():
            raise ValueError("verdict_summary must not be empty")
        if not self.rationale.strip():
            raise ValueError("rationale must not be empty")
        self.cited_artifact_paths = list(dict.fromkeys(self.cited_artifact_paths))
        if not self.cited_artifact_paths:
            raise ValueError("artifact citation is required")
        return self


class UpliftAdvisoryReport(BaseModel):
    """Strict stakeholder narrative grounded in deterministic artifact paths."""

    model_config = ConfigDict(extra="forbid")

    title: str
    executive_summary: str
    validation_summary: str
    held_out_summary: str
    scoring_summary: str
    limitations: List[str] = Field(default_factory=list)
    cited_artifact_paths: List[str]

    @model_validator(mode="after")
    def _validate_report(self) -> "UpliftAdvisoryReport":
        required_text = [
            self.title,
            self.executive_summary,
            self.validation_summary,
            self.held_out_summary,
            self.scoring_summary,
        ]
        if any(not value.strip() for value in required_text):
            raise ValueError("report text fields must not be empty")
        self.limitations = [
            limitation.strip()
            for limitation in self.limitations
            if limitation.strip()
        ]
        self.cited_artifact_paths = list(dict.fromkeys(self.cited_artifact_paths))
        if not self.cited_artifact_paths:
            raise ValueError("artifact citation is required")
        return self


class UpliftResultCard(BaseModel):
    """Result object for one uplift trial.

    The top-level ``qini_auc`` / ``uplift_auc`` / ``uplift_at_k`` /
    ``policy_gain`` fields are the *validation* metrics used to select a
    champion across trials. The optional ``held_out_*`` fields carry an
    honest held-out test estimate for the same fitted model. Champion
    selection MUST use the validation fields; reporting SHOULD prefer the
    held-out fields when present.
    """

    result_id: str = Field(default_factory=lambda: f"UR-{uuid4().hex[:6]}")
    trial_spec_id: str
    status: Literal["success", "failed", "partial"] = "success"
    error: Optional[str] = None
    qini_auc: Optional[float] = None
    uplift_auc: Optional[float] = None
    uplift_at_k: Dict[str, Optional[float]] = Field(default_factory=dict)
    policy_gain: Dict[str, Optional[float]] = Field(default_factory=dict)
    held_out_qini_auc: Optional[float] = None
    held_out_uplift_auc: Optional[float] = None
    held_out_uplift_at_k: Dict[str, Optional[float]] = Field(default_factory=dict)
    held_out_policy_gain: Dict[str, Optional[float]] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    artifact_paths: Dict[str, str] = Field(default_factory=dict)


class UpliftExperimentRecord(BaseModel):
    """Append-only ledger record for an uplift experiment."""

    run_id: str = Field(default_factory=lambda: f"RUN-{uuid4().hex[:8]}")
    parent_run_id: Optional[str] = None
    hypothesis_id: str
    feature_recipe_id: str
    feature_artifact_id: str = ""
    template_name: str = ""
    uplift_learner_family: str
    base_estimator: str
    params_hash: str
    split_seed: int
    status: Literal["success", "failed", "partial"] = "success"
    error: Optional[str] = None
    qini_auc: Optional[float] = None
    uplift_auc: Optional[float] = None
    uplift_at_k: Dict[str, Optional[float]] = Field(default_factory=dict)
    policy_gain: Dict[str, Optional[float]] = Field(default_factory=dict)
    held_out_qini_auc: Optional[float] = None
    held_out_uplift_auc: Optional[float] = None
    held_out_uplift_at_k: Dict[str, Optional[float]] = Field(default_factory=dict)
    held_out_policy_gain: Dict[str, Optional[float]] = Field(default_factory=dict)
    verdict: Literal["supported", "refuted", "inconclusive", "baseline"] = "baseline"
    next_recommended_actions: List[str] = Field(default_factory=list)
    artifact_paths: Dict[str, str] = Field(default_factory=dict)


class UpliftSubmissionArtifact(BaseModel):
    """Validated final scoring artifact for unlabeled uplift_test.csv rows."""

    artifact_path: str
    champion_trial_id: str
    feature_recipe_id: str
    feature_artifact_id: str
    row_count: int
    columns: List[str] = Field(default_factory=lambda: ["client_id", "uplift"])
    scoring_table: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)

    @model_validator(mode="after")
    def _validate_schema(self) -> "UpliftSubmissionArtifact":
        # Schema is exactly two columns: an entity key (whatever the contract declares)
        # plus the literal "uplift" score. The contract-aware
        # validate_submission_artifact() in reporting.py enforces the entity key
        # name; this model just ensures the shape.
        if len(self.columns) != 2 or self.columns[-1] != "uplift":
            raise ValueError(
                "submission columns must be exactly [<entity_key>, 'uplift']"
            )
        if self.row_count < 0:
            raise ValueError("row_count must be non-negative")
        return self


class UpliftProjectContract(BaseModel):
    """Source-of-truth contract for a campaign-uplift modeling task."""

    schema_version: str = "1.0"
    project_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    task_name: str
    description: str = ""

    table_schema: UpliftTableSchema
    entity_key: str = "client_id"
    treatment_column: str = "treatment_flg"
    target_column: str = "target"
    scoring_uplift_column: str = "uplift"
    submission_policy: Literal["scoring_only"] = "scoring_only"

    evaluation_policy: UpliftEvaluationPolicy = Field(
        default_factory=UpliftEvaluationPolicy
    )
    split_contract: UpliftSplitContract = Field(default_factory=UpliftSplitContract)
    feature_sources: List[str] = Field(default_factory=lambda: ["clients", "purchases"])

    @model_validator(mode="after")
    def _validate_semantic_columns(self) -> "UpliftProjectContract":
        semantic_columns = {
            self.entity_key,
            self.treatment_column,
            self.target_column,
            self.scoring_uplift_column,
        }
        if len(semantic_columns) != 4:
            raise ValueError(
                "entity_key, treatment_column, target_column, and scoring_uplift_column must be distinct"
            )
        return self
