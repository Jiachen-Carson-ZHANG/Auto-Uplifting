"""Reporting and scoring artifacts for uplift experiments."""
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from src.models.uplift import (
    UpliftExperimentRecord,
    UpliftFeatureArtifact,
    UpliftProjectContract,
    UpliftSubmissionArtifact,
    UpliftTrialSpec,
)
from src.uplift.templates import FittedUpliftModel


def _is_finite(value: float | None) -> bool:
    return value is not None and not (isinstance(value, float) and value != value)  # noqa: PLR0124


def _champion(records: List[UpliftExperimentRecord]) -> UpliftExperimentRecord | None:
    successful = [record for record in records if record.status == "success"]
    if not successful:
        return None
    return max(
        successful,
        key=lambda record: record.qini_auc if _is_finite(record.qini_auc) else float("-inf"),
    )


def generate_uplift_report(
    contract: UpliftProjectContract,
    *,
    records: List[UpliftExperimentRecord],
    output_path: str | Path,
) -> str:
    """Write a concise stakeholder-facing markdown report grounded in ledger records."""
    champion = _champion(records)
    lines = [
        "# Uplift Experiment Report",
        "",
        f"Task: {contract.task_name}",
        "",
        "Internal evaluation uses labeled uplift_train.csv splits.",
        "uplift_test.csv is scoring/submission only and is not used for metrics.",
        "",
        "## Evaluation protocol",
        "- Selection metrics (Qini AUC, Uplift AUC, Uplift@k, policy gain) are computed",
        "  on the validation partition and drive champion selection across trials.",
        "- Held-out metrics (when reported) come from the same fitted model scored on a",
        "  test partition that was not used for selection. They are the honest",
        "  generalization estimate for that fit.",
        "- The submission CSV is produced by a model retrained on the full labeled",
        "  training set. That model has seen the validation and test rows, so its true",
        "  performance on unlabeled customers may differ from the held-out estimate.",
        "",
        "## Champion",
    ]
    if champion is None:
        lines.extend(["No successful trial is available yet.", ""])
    else:
        lines.extend(
            [
                f"Template: {champion.template_name}",
                f"Hypothesis: {champion.hypothesis_id}",
                "",
                "### Validation (selection)",
                f"- Qini AUC: {champion.qini_auc}",
                f"- Uplift AUC: {champion.uplift_auc}",
                f"- Uplift@k: {champion.uplift_at_k}",
                f"- Policy gain: {champion.policy_gain}",
            ]
        )
        if _is_finite(champion.held_out_qini_auc) or champion.held_out_uplift_at_k:
            lines.extend(
                [
                    "",
                    "### Held-out test (honest generalization estimate)",
                    f"- Qini AUC: {champion.held_out_qini_auc}",
                    f"- Uplift AUC: {champion.held_out_uplift_auc}",
                    f"- Uplift@k: {champion.held_out_uplift_at_k}",
                    f"- Policy gain: {champion.held_out_policy_gain}",
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    "### Held-out test",
                    "- Not available: no test partition was configured.",
                ]
            )
        lines.append("")

    lines.append("## Trial Ledger")
    for record in records:
        lines.append(
            "- "
            f"{record.template_name}: status={record.status}, "
            f"val_qini_auc={record.qini_auc}, val_uplift_auc={record.uplift_auc}, "
            f"test_qini_auc={record.held_out_qini_auc}, "
            f"hypothesis={record.hypothesis_id}"
        )

    lines.extend(
        [
            "",
            "## Limitations",
            "- Response-model baselines rank purchase propensity and are not true uplift learners.",
            "- Campaign cost recommendations are sensitivity-based unless real communication cost is configured.",
            "- Uplift@k is undefined (NaN) when the top-k slice has zero treated or zero control rows; the report surfaces the NaN rather than a misleading 0.",
            "- Final submission scores are rankings for unlabeled customers, not evaluation metrics.",
        ]
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def generate_submission_artifact(
    contract: UpliftProjectContract,
    *,
    model: FittedUpliftModel,
    scoring_feature_artifact: UpliftFeatureArtifact,
    champion_trial: UpliftTrialSpec,
    output_path: str | Path,
) -> UpliftSubmissionArtifact:
    """Score unlabeled uplift_test rows and write client_id,uplift submission CSV."""
    scoring_features = pd.read_csv(scoring_feature_artifact.artifact_path)
    forbidden = {contract.target_column, contract.treatment_column}
    present_forbidden = sorted(forbidden.intersection(scoring_features.columns))
    if present_forbidden:
        raise ValueError(f"scoring features contain forbidden columns: {present_forbidden}")

    submission = pd.DataFrame(
        {
            contract.entity_key: scoring_features[contract.entity_key],
            "uplift": model.predict_uplift(scoring_features),
        }
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False)

    return UpliftSubmissionArtifact(
        artifact_path=str(path),
        champion_trial_id=champion_trial.spec_id,
        feature_recipe_id=scoring_feature_artifact.feature_recipe_id,
        feature_artifact_id=scoring_feature_artifact.feature_artifact_id,
        row_count=len(submission),
        columns=submission.columns.tolist(),
        scoring_table=contract.table_schema.scoring_table,
    )


def validate_submission_artifact(
    contract: UpliftProjectContract,
    artifact: UpliftSubmissionArtifact,
) -> None:
    """Validate submission schema and exact scoring customer coverage."""
    submission = pd.read_csv(artifact.artifact_path)
    scoring_ids = pd.read_csv(contract.table_schema.scoring_table, usecols=[contract.entity_key])
    expected_columns = [contract.entity_key, "uplift"]

    if submission.columns.tolist() != expected_columns:
        raise ValueError(f"submission columns must be exactly {expected_columns}")
    if len(submission) != len(scoring_ids):
        raise ValueError("submission row count does not match scoring table")
    if submission[contract.entity_key].duplicated().any():
        raise ValueError(f"submission has duplicate {contract.entity_key} rows")
    if set(submission[contract.entity_key]) != set(scoring_ids[contract.entity_key]):
        raise ValueError(
            f"submission {contract.entity_key} set does not match scoring table"
        )
    if not pd.api.types.is_numeric_dtype(submission["uplift"]):
        raise ValueError("submission uplift column must be numeric")
