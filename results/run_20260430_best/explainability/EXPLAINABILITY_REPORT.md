# AutoLift Explainability Pack

This pack adapts the visual explanation structure from `human_baseline_uplift.ipynb` for the AutoLift run. It adds model-performance visuals, targeting diagnostics, feature-level explanation, and an agent decision timeline.

## Human vs AutoLift Pending Re-Audit

The human-baseline workflow is being corrected separately. The table and SVG
below preserve the previously observed notebook numbers for traceability only;
do not use them as final win/loss claims.

| Metric | AutoLift Retrospective Reference | Provisional Human Notebook Row | Provisional Delta |
| --- | --- | --- | --- |
| Held-out raw Qini AUC | 331.7694 | 328.3899 | +3.3795 |
| Held-out uplift AUC | 0.06149 | 0.0631 | -0.00161 |
| Held-out uplift@5% | 0.139969 | 0.1637 | -0.023731 |
| Held-out uplift@10% | 0.099709 | 0.1289 | -0.029191 |
| Held-out uplift@20% | 0.071709 | 0.0764 | -0.004691 |
| Held-out uplift@30% | 0.062456 | 0.0627 | -0.000244 |

![Held-out top-k comparison](human_vs_autolift_topk.svg)

The comparison graphic is provisional until the corrected human-baseline run is
available. The stable claim is about AutoLift's auditable workflow and reasoning
trail, not a finalized metric win over the human baseline.

## AutoLift Curves

![Held-out Qini curve](autolift_heldout_qini_curve.svg)

![Held-out uplift curve](autolift_heldout_uplift_curve.svg)

The notebook did not leave prediction-level human artifacts in this workspace, so
the provisional human comparison is shown through notebook-reported metrics
instead of an overlaid human curve.

## Decile Lift

![Held-out decile lift](autolift_decile_lift.svg)

The first decile is the top predicted-uplift group and should show the clearest treatment/control response separation if the ranking is useful.

## Feature Explanation

![XAI top drivers](autolift_xai_top_drivers.svg)

| Rank | Feature | Mean Abs Uplift Change | Direction |
| --- | --- | --- | --- |
| 1 | age_clean | 0.013934 | higher_feature_higher_uplift |
| 2 | days_to_first_redeem | 0.013803 | higher_feature_lower_uplift |
| 3 | points_received_total_30d | 0.009384 | higher_feature_lower_uplift |
| 4 | purchase_sum_60d | 0.009088 | higher_feature_lower_uplift |
| 5 | account_age_days | 0.008941 | higher_feature_lower_uplift |
| 6 | purchase_sum_90d | 0.006571 | higher_feature_lower_uplift |
| 7 | basket_quantity_30d | 0.006465 | higher_feature_lower_uplift |
| 8 | points_spent_to_purchase_ratio_lifetime | 0.006298 | higher_feature_lower_uplift |
| 9 | avg_transaction_value_30d | 0.005967 | higher_feature_lower_uplift |
| 10 | points_received_total_90d | 0.005600 | higher_feature_lower_uplift |

Age-related features remain prominent, so this explanation should be presented with the feature-policy caveat already documented in the robustness audit.

## Agent Reasoning Timeline

![Agent reasoning timeline](agent_reasoning_timeline.svg)

| Step | Run | Learner | Estimator | Held-out Qini | Verdict | Decision Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | RUN-447718f5 | two_model | logistic_regression | 309.7954 | baseline |  |
| 2 | RUN-9c2b8311 | class_transformation | gradient_boosting | 294.6742 | inconclusive | The trial's hold-out uplift-AUC fell from 0.04667 (baseline) to 0.04245 and Qini-AUC from 309.80 to 294.67, so it fails to beat the existing model. |
| 3 | RUN-bfd6fa1c | two_model | xgboost | 310.2122 | supported | The trial's held-out uplift AUC rose from 0.0467 to 0.0580 (+24%), and Qini AUC edged up from 309.8 to 310.2, with uplift@5% improving to 0.1315. |
| 4 | RUN-f7bdb1dc | class_transformation | xgboost | 325.9836 | supported | The new model outperforms the prior champion on held-out uplift metrics: Qini AUC rises from 310.21 to 325.98 and uplift-AUC from 0.058026 to 0.058... |
| 5 | RUN-dd10fc91 | two_model | lightgbm | 300.1975 | inconclusive | The new model's held-out Qini AUC dropped from 325.98 to 300.20 and uplift-AUC fell from 0.0589 to 0.0563 versus the prior champion, missing the ta... |
| 6 | RUN-c5e6e86f | class_transformation | lightgbm | 331.7694 | supported | The LightGBM model improved held-out Qini AUC from 325.98 to 331.77 (up 5.79) and uplift-AUC from 0.05888 to 0.06149 (up 0.00261), both exceeding t... |

This is the agent-specific contribution: each trial carries a hypothesis, feature rationale, expected signal, held-out metrics, judge verdict, XAI summary, and policy recommendation.

## Source Notes

- AutoLift artifacts: `results/run_20260430_best/uplift_ledger.jsonl` and saved champion CSVs under `artifacts/uplift/run_20260430_221602/runs/UT-9fb6c6/`.
- Human metrics: `human_baseline_uplift.ipynb` outputs. The best held-out row is `class_transform_gbm` / `tuned_class_transform_gbm`; the validation-selected champion is `tuned_solo_model_xgb`.
- Metric caution: AutoLift report-table Qini values are normalized, while the human notebook reports raw Qini. This pack compares raw Qini only where both sides expose raw Qini.
