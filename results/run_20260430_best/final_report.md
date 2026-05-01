# AutoLift Experiment Report

Task: retailhero-uplift

## Decision

Use `RUN-f1c30175` as the leakage-clean validation+CV selected AutoLift candidate. Selection was made without internal test/held-out metrics: first rank by validation predictions, then rerank the top 3 with 5-fold CV over the original train+validation pool only. The internal test partition stayed sealed until the final audit.

Important boundary: `RUN-c5e6e86f` remains the best retrospective held-out main-run row, but it must not be described as the leakage-clean selected champion because the earlier champion logic considered held-out performance. The human-baseline comparison is pending re-audit because the human notebook workflow is being corrected separately; do not use current human deltas as final claims.

## Leakage-Clean Validation+CV Selection

- Run ID: RUN-f1c30175
- Source spec: agentic_tune__UT-bc7585__p14
- Template: two_model_lightgbm
- Learner family: two_model
- Base estimator: lightgbm
- Validation raw Qini AUC: 353.426059
- Validation Normalized Qini AUC: 0.426365
- Validation Uplift AUC: 0.066839
- CV mean Normalized Qini AUC: 0.396226
- CV std Normalized Qini AUC: 0.060313
- CV mean Uplift AUC: 0.066282
- CV mean Uplift@5%: 0.143441
- CV mean Uplift@10%: 0.115764
- Sealed held-out raw Qini AUC: 309.987113
- Sealed held-out Normalized Qini AUC: 0.248455
- Sealed held-out Uplift AUC: 0.058746
- Sealed held-out Uplift@5%: 0.183569
- Sealed held-out Uplift@10%: 0.111772
- Sealed held-out Uplift@20%: 0.064553
- Sealed held-out Uplift@30%: 0.058085

CV artifact: `validation_top3_cv_audit.md`

## Retrospective Held-Out Best Reference

This row is useful for diagnosis and comparison, but not for leakage-clean champion selection.

- Run ID: RUN-c5e6e86f
- Source spec: UT-9fb6c6
- Template: class_transformation_lightgbm
- Held-out raw Qini AUC: 331.769404
- Held-out Normalized Qini AUC: 0.337369
- Held-out Uplift AUC: 0.06149
- Held-out Uplift@5%: 0.139969
- Held-out Uplift@10%: 0.099709
- Held-out Uplift@20%: 0.071709
- Held-out Uplift@30%: 0.062456

## Human Notebook Benchmark

Source: `human_baseline_uplift.ipynb`

Status: pending re-audit. The values below are retained only as the previously
observed notebook outputs and must not be treated as final comparison numbers
until the corrected human-baseline workflow is pushed and rerun.

Previously observed best held-out human notebook row: `class_transform_gbm` / `tuned_class_transform_gbm`

- Held-out Qini AUC: 328.3899
- Held-out Uplift AUC: 0.0631
- Held-out Uplift@5%: 0.1637
- Held-out Uplift@10%: 0.1289
- Held-out Uplift@20%: 0.0764
- Held-out Uplift@30%: 0.0627

Previously observed notebook-selected validation champion: `tuned_solo_model_xgb`

- Validation Qini AUC: 367.03263
- Held-out Test Qini AUC: 299.13056
- Held-out Test Uplift AUC: 0.05828
- Held-out Test Uplift@10%: 0.12779
- Held-out Test Uplift@30%: 0.05242

For slide/report comparison, treat the human-baseline column as pending. The
notebook-selected validation champion is listed for traceability, but the final
human benchmark should come from the corrected human-baseline run.

## Internal AutoLift Reference

- Run ID: RUN-447718f5
- Template: two_model_sklearn
- Normalized Qini AUC: 0.193438
- Uplift AUC: 0.044687
- Held-out Normalized Qini AUC: 0.247670
- Held-out Uplift AUC: 0.046667

## All Trials

| Run | Role | Learner | Estimator | Val Normalized Qini | Val Uplift AUC | Held-out Normalized Qini | Held-out Uplift AUC |
|---|---|---|---|---:|---:|---:|---:|
| RUN-447718f5 | Internal reference | two_model | logistic_regression | 0.193438 | 0.044687 | 0.247670 | 0.046667 |
| RUN-9c2b8311 | Agent | class_transformation | gradient_boosting | 0.260972 | 0.044990 | 0.186826 | 0.042445 |
| RUN-bfd6fa1c | Agent | two_model | xgboost | 0.408278 | 0.064369 | 0.249374 | 0.058026 |
| RUN-f7bdb1dc | Agent | class_transformation | xgboost | 0.377303 | 0.060407 | 0.313751 | 0.058876 |
| RUN-dd10fc91 | Agent | two_model | lightgbm | 0.407684 | 0.065052 | 0.208495 | 0.056286 |
| RUN-c5e6e86f | Agent | class_transformation | lightgbm | 0.344842 | 0.060678 | 0.337369 | 0.061490 |

## Human vs AutoLift Comparison Pending

The AutoLift side below is final for this audit. The human side is provisional
and must be refreshed after the corrected human-baseline code is pushed.

Strict leakage-clean selection versus previously observed human numbers:

| Metric | AutoLift Validation+CV Candidate | Provisional Human Notebook Row | Provisional Delta |
|---|---:|---:|---:|
| Held-out raw Qini AUC | 309.9871 | 328.3899 | -18.4028 |
| Held-out uplift AUC | 0.058746 | 0.0631 | -0.004354 |
| Held-out uplift@5% | 0.183569 | 0.1637 | +0.019869 |
| Held-out uplift@10% | 0.111772 | 0.1289 | -0.017128 |
| Held-out uplift@20% | 0.064553 | 0.0764 | -0.011847 |
| Held-out uplift@30% | 0.058085 | 0.0627 | -0.004615 |

Retrospective held-out best reference versus previously observed human numbers:

| Metric | AutoLift Retrospective Best | Provisional Human Notebook Row | Provisional Delta |
|---|---:|---:|---:|
| Held-out raw Qini AUC | 331.7694 | 328.3899 | +3.3795 |
| Held-out uplift AUC | 0.06149 | 0.0631 | -0.00161 |
| Held-out uplift@5% | 0.139969 | 0.1637 | -0.023731 |
| Held-out uplift@10% | 0.099709 | 0.1289 | -0.029191 |
| Held-out uplift@20% | 0.071709 | 0.0764 | -0.004691 |
| Held-out uplift@30% | 0.062456 | 0.0627 | -0.000244 |

Interpretation: do not claim a final win/loss against the human baseline until
the corrected human-baseline run is available. The current defensible claim is
about AutoLift's own process: it produced an auditable end-to-end agentic
workflow, generated candidate families and tuning plans, detected its own
leakage risk, and preserved reasoning and artifacts for review.

## Agentic Tuning Audit

- Plan artifact: `agentic_tuning_plan.json`
- Execution summary: `agentic_tuning_execution_summary.json`
- Tuning ledger: `agentic_tuning_ledger.jsonl`
- Validation-only audit plan: `agentic_tuning_plan_validation_only.json`
- Validation-only audit summary: `agentic_tuning_validation_only_execution_summary.json`
- Validation-only audit ledger: `agentic_tuning_validation_only_ledger.jsonl`
- Validation top-3 CV audit: `validation_top3_cv_audit.md`
- Validation top-3 CV leaderboard: `validation_top3_cv_leaderboard.csv`
- Tuning seed: 20260501
- Trial count: 32 / 32 successful
- Candidate families tuned: `class_transformation_lightgbm`, `class_transformation_xgboost`
- Tuned XAI summary: `tuned_xai_summary.json`

Audit finding: the original tuning loop did not use the human notebook benchmark,
but its selector could see internal held-out metrics. That makes the tuned
RUN-2af274da result invalid for final champion claims. A patched validation-only
tuning rerun selected `RUN-d97c36d3` / `two_model_xgboost` by validation raw
Qini (357.375156). A stricter follow-up CV audit reranked the top 3 validation
candidates using only the original train+validation pool and selected
`RUN-f1c30175` / `two_model_lightgbm` by mean normalized CV Qini. Only after
that selection was the sealed internal test audit opened, where `RUN-f1c30175`
scored 309.987113 raw Qini. Therefore `RUN-f1c30175` is the leakage-clean
selection result, while `RUN-c5e6e86f` is retained only as a retrospective
held-out-best reference.

## Explainability Pack

Visual explanation assets are available in `explainability/EXPLAINABILITY_REPORT.md`.

The current visual pack was generated for `RUN-c5e6e86f`, the retrospective
held-out-best reference. It is useful for explaining the agent workflow and
model behavior, but should not be presented as the XAI pack for the strict
validation+CV selected candidate `RUN-f1c30175`.

The pack adapts the human notebook's explanation style for AutoLift:

- Human-vs-AutoLift top-k targeting comparison.
- AutoLift held-out Qini and uplift curves.
- AutoLift held-out decile lift chart.
- Champion XAI top-driver chart.
- Agent reasoning timeline showing hypotheses, verdicts, and metric evidence across trials.

Use this pack to support the agent contribution: AutoLift is not only a model score, but an auditable end-to-end experimentation loop.

## Seed Stability

No repeated-seed stability groups are available yet.

## Feature Semantics

| Feature Recipe ID | Temporal Policy | Best Held-out Normalized Qini | Intended Signal | XAI Check |
|---|---|---:|---|---|
| 0b2e3552e7bd | safe_history_until_reference | 0.337369 | Held-out Qini should stay at or above 326 and uplift-AUC ≥0.060; XAI top drivers should be recency, frequency, points... | age_dominance_warning=True; behavioral_top5_present=True |
| 4f0cb0168773 | post_issue_history | 0.186826 | Higher uplift-AUC and a Qini curve showing separation driven by behavioural recency, points burn-rate and basket dept... | age_dominance_warning=True; behavioral_top5_present=True |

## Policy Recommendation

This policy block was generated from `RUN-c5e6e86f` artifacts. Treat it as a
retrospective policy example unless a matching policy/XAI pack is regenerated
for `RUN-f1c30175`.

- Recommended threshold: 5%
- Rationale: The 5% cutoff delivers the highest lift rate (14%) and a strong positive ROI (0.40), while aligning with the elbow threshold to focus on top responders under budget.
- Segment summary: {'total_customers': 30006, 'persuadables': 3001, 'sleeping_dogs': 3001, 'middle_ground': 24004, 'persuadable_pct': 0.1, 'sleeping_dog_pct': 0.1}
- First targeting cutoff: {'threshold_pct': 5, 'n_targeted': 1501, 'lift_rate': 0.14, 'incremental_conversions': 210.09, 'total_cost': 1501.0, 'estimated_revenue_lift': 2100.93, 'roi': 0.4}

## Explanation

- Method: cached_model_permutation
- Top drivers: [{'feature': 'age_clean', 'mean_abs_uplift_change': 0.013934, 'spearman_with_uplift': 0.3724, 'direction': 'higher_feature_higher_uplift'}, {'feature': 'days_to_first_redeem', 'mean_abs_uplift_change': 0.013803, 'spearman_with_uplift': -0.177, 'direction': 'higher_feature_lower_uplift'}, {'feature': 'points_received_total_30d', 'mean_abs_uplift_change': 0.009384, 'spearman_with_uplift': -0.2579, 'direction': 'higher_feature_lower_uplift'}, {'feature': 'purchase_sum_60d', 'mean_abs_uplift_change': 0.009088, 'spearman_with_uplift': -0.3672, 'direction': 'higher_feature_lower_uplift'}, {'feature': 'account_age_days', 'mean_abs_uplift_change': 0.008941, 'spearman_with_uplift': -0.2089, 'direction': 'higher_feature_lower_uplift'}, {'feature': 'purchase_sum_90d', 'mean_abs_uplift_change': 0.006571, 'spearman_with_uplift': -0.3573, 'direction': 'higher_feature_lower_uplift'}, {'feature': 'basket_quantity_30d', 'mean_abs_uplift_change': 0.006465, 'spearman_with_uplift': -0.2992, 'direction': 'higher_feature_lower_uplift'}, {'feature': 'points_spent_to_purchase_ratio_lifetime', 'mean_abs_uplift_change': 0.006298, 'spearman_with_uplift': -0.1781, 'direction': 'higher_feature_lower_uplift'}, {'feature': 'avg_transaction_value_30d', 'mean_abs_uplift_change': 0.005967, 'spearman_with_uplift': -0.1642, 'direction': 'higher_feature_lower_uplift'}, {'feature': 'points_received_total_90d', 'mean_abs_uplift_change': 0.0056, 'spearman_with_uplift': -0.3222, 'direction': 'higher_feature_lower_uplift'}]
- Leakage flag: True

## Representative Cases

{'highest_uplift': [{'client_id': 'bb32a9bb43', 'uplift': 0.19905557734915647, 'age_clean': 65.0, 'days_to_first_redeem': 552.0, 'points_received_total_30d': 2.0}, {'client_id': 'cc66ab8b43', 'uplift': 0.19582031194238403, 'age_clean': 53.0, 'days_to_first_redeem': 23.0, 'points_received_total_30d': 39.6}, {'client_id': '2d594e4f54', 'uplift': 0.188900684734439, 'age_clean': 62.0, 'days_to_first_redeem': 82.0, 'points_received_total_30d': 1.9}], 'lowest_uplift': [{'client_id': '7666193e73', 'uplift': -0.04241989947605418, 'age_clean': 58.0, 'days_to_first_redeem': 815.0, 'points_received_total_30d': 134.4}, {'client_id': 'ecd5de5b27', 'uplift': -0.04715497567508797, 'age_clean': 47.0, 'days_to_first_redeem': 48.0, 'points_received_total_30d': 14.6}, {'client_id': '80ddf8fbfc', 'uplift': -0.0627210953228261, 'age_clean': 37.0, 'days_to_first_redeem': 367.0, 'points_received_total_30d': 0.4}], 'near_boundary': [{'client_id': 'e31d0c5f5d', 'uplift': 0.028971261981808993, 'age_clean': 75.0, 'days_to_first_redeem': 413.0, 'points_received_total_30d': 22.2}, {'client_id': '5cb106286b', 'uplift': 0.0289673487489297, 'age_clean': 40.0, 'days_to_first_redeem': 52.0, 'points_received_total_30d': 12.6}, {'client_id': 'f5c601a81c', 'uplift': 0.028844802454976337, 'age_clean': 46.0, 'days_to_first_redeem': 65.0, 'points_received_total_30d': 49.1}]}

## Hypothesis Loop

- Judge verdict: supported
- Metric evidence: {'qini_auc': 331.769404, 'normalized_qini_auc': 0.337369, 'uplift_auc': 0.06149, 'uplift_at_5pct': 0.139969, 'uplift_at_10pct': 0.099709, 'uplift_at_20pct': 0.071709, 'evaluation_surface': 'held_out'}
- Key evidence: ['Held-out Qini AUC: 325.9836 → 331.7694', 'Held-out uplift-AUC: 0.058876 → 0.06149']
- New hypothesis suggestion: None

## Retry Decision

- Continue: False
- Reason: Reached max_trials=5.
- Next action: generate_report

## Why this answer is credible

- Human notebook benchmark is reported separately from the internal AutoLift reference run.
- AutoLift trial-table Qini values are normalized by the report-facing perfect-oracle Qini denominator; the human notebook comparison uses raw Qini because `human_baseline_uplift.ipynb` reports raw Qini.
- Trial records come from the append-only uplift ledger.
- Policy recommendations are derived from saved uplift_scores.csv artifacts.
- XAI is used as supporting evidence for hypothesis explanation, not as the metric source of truth.

## Trial Count

6 ledger records.
