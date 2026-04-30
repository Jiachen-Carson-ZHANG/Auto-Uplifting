# AutoLift Experiment Report

Task: retailhero-uplift

## Decision

Use agent champion RUN-c5e6e86f as the AutoLift champion. Against the real human notebook benchmark (`human_baseline_uplift.ipynb`), this is a narrow held-out Qini win, not an across-the-board win: AutoLift held-out raw Qini is 331.7694 vs the human notebook best held-out row at 328.3899 (+3.3795). The human notebook best is stronger on held-out uplift AUC and top-k lift at 5%, 10%, and 20%.

## Agent Champion

- Run ID: RUN-c5e6e86f
- Template: class_transformation_lightgbm
- Learner family: class_transformation
- Base estimator: lightgbm
- Normalized Qini AUC: 0.344842
- Uplift AUC: 0.060678
- Held-out Normalized Qini AUC: 0.337369
- Held-out Uplift AUC: 0.06149
- Policy gain: {'top_5pct_zero_cost': 211.258245, 'top_5pct_low_cost': 136.208245, 'top_5pct_medium_cost': -88.941755, 'top_10pct_zero_cost': 271.206372, 'top_10pct_low_cost': 121.156372, 'top_10pct_medium_cost': -328.993628, 'top_20pct_zero_cost': 386.336736, 'top_20pct_low_cost': 86.236736, 'top_20pct_medium_cost': -814.063264, 'top_30pct_zero_cost': 478.888396, 'top_30pct_low_cost': 28.788396, 'top_30pct_medium_cost': -1321.511604}

## Human Notebook Benchmark

Source: `human_baseline_uplift.ipynb`

Best held-out human notebook row: `class_transform_gbm` / `tuned_class_transform_gbm`

- Held-out Qini AUC: 328.3899
- Held-out Uplift AUC: 0.0631
- Held-out Uplift@5%: 0.1637
- Held-out Uplift@10%: 0.1289
- Held-out Uplift@20%: 0.0764
- Held-out Uplift@30%: 0.0627

Notebook-selected validation champion: `tuned_solo_model_xgb`

- Validation Qini AUC: 367.03263
- Held-out Test Qini AUC: 299.13056
- Held-out Test Uplift AUC: 0.05828
- Held-out Test Uplift@10%: 0.12779
- Held-out Test Uplift@30%: 0.05242

For slide/report comparison, use the best held-out human notebook row above. The notebook-selected validation champion is listed for traceability, but it is not the strongest human held-out test result.

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

## Human vs AutoLift Comparison

| Metric | AutoLift Champion | Human Notebook Best Held-out Row | Delta |
|---|---:|---:|---:|
| Held-out raw Qini AUC | 331.7694 | 328.3899 | +3.3795 |
| Held-out uplift AUC | 0.06149 | 0.0631 | -0.00161 |
| Held-out uplift@5% | 0.139969 | 0.1637 | -0.023731 |
| Held-out uplift@10% | 0.099709 | 0.1289 | -0.029191 |
| Held-out uplift@20% | 0.071709 | 0.0764 | -0.004691 |
| Held-out uplift@30% | 0.062456 | 0.0627 | -0.000244 |

Interpretation: AutoLift is the held-out Qini leader, but the real human notebook remains competitive and is stronger on several targeting lift metrics. Present the result as a marginal Qini improvement from automation, not as a broad dominance claim.

## Explainability Pack

Visual explanation assets are available in `explainability/EXPLAINABILITY_REPORT.md`.

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
