# Final Honest Human vs AutoLift CV Comparison

Both sides below use a leakage-clean selection boundary:

- Human baseline: `human_baseline_uplift.ipynb` screens models on the original validation split, reranks the top 3 non-random candidates with 5-fold CV over train+validation, selects `solo_model_xgb` by CV mean normalized Qini, then opens the sealed test set.
- AutoLift: selects the validation top 3 without held-out metrics, reranks them with 5-fold CV over train+validation, selects `RUN-f1c30175` / `two_model_lightgbm` by CV mean normalized Qini, then opens the sealed test set.

## Selected Models

| System | Selected model | Selection metric | CV mean norm Qini | CV std norm Qini | Sealed test norm Qini |
|---|---|---|---:|---:|---:|
| AutoLift | `RUN-f1c30175` / `two_model_lightgbm` | CV mean normalized Qini | 0.396226 | 0.060313 | 0.248455 |
| Human baseline | `solo_model_xgb` | CV mean normalized Qini | 0.409490 | 0.087550 | 0.204120 |

## CV Comparison

| Metric | AutoLift CV | Human CV | AutoLift - Human |
|---|---:|---:|---:|
| Normalized Qini AUC | 0.396226 | 0.409490 | -0.013264 |
| Normalized Qini AUC std | 0.060313 | 0.087550 | -0.027237 |
| Raw Qini AUC | 392.456261 | 396.146460 | -3.690199 |
| Uplift AUC | 0.066282 | 0.065320 | +0.000962 |
| Uplift@5% | 0.143441 | 0.133600 | +0.009841 |
| Uplift@10% | 0.115764 | 0.110320 | +0.005444 |
| Uplift@30% | 0.064839 | 0.070410 | -0.005571 |

## Sealed Test Comparison

| Metric | AutoLift test | Human test | AutoLift - Human |
|---|---:|---:|---:|
| Normalized Qini AUC | 0.248455 | 0.204120 | +0.044335 |
| Raw Qini AUC | 309.987113 | 299.125590 | +10.861523 |
| Uplift AUC | 0.058746 | 0.057820 | +0.000926 |
| Uplift@5% | 0.183569 | 0.155200 | +0.028369 |
| Uplift@10% | 0.111772 | 0.092310 | +0.019462 |
| Uplift@30% | 0.058085 | 0.051870 | +0.006215 |

## Interpretation

The human baseline is slightly ahead on the CV selection metric, while AutoLift is more stable across folds and generalizes better on the sealed test metrics available here. The fair final statement is:

AutoLift does not dominate the human workflow on CV mean normalized Qini, but its leakage-clean CV-selected candidate beats the honest human CV-selected champion on sealed test normalized Qini, raw Qini, uplift AUC, and top-k lift at 5%, 10%, and 30%.

This comparison should replace the earlier retrospective held-out-best comparison.
