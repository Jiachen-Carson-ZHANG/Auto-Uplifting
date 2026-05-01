# Validation Top-K Cross-Validation Audit

Candidates are selected from validation predictions only. The internal test partition is excluded from CV and remains a final audit surface.

- Top-k: 3
- Folds: 5
- Seed: `20260501`
- Selected by CV: `RUN-f1c30175` / `agentic_tune__UT-bc7585__p14`

## CV Leaderboard

| Validation Rank | Run | Template | Val Norm Qini | CV Mean Norm Qini | CV Std Norm Qini | CV Mean Uplift@10% |
|---:|---|---|---:|---:|---:|---:|
| 3 | `RUN-f1c30175` | `two_model_lightgbm` | 0.426365 | 0.396226 | 0.060313 | 0.115764 |
| 1 | `RUN-d97c36d3` | `two_model_xgboost` | 0.442493 | 0.378029 | 0.031675 | 0.104811 |
| 2 | `RUN-5783a863` | `two_model_xgboost` | 0.427435 | 0.349423 | 0.042535 | 0.109287 |

## Split Boundary

- full_labeled_rows: `200039`
- cv_pool_rows: `170033`
- internal_train_rows: `140027`
- internal_validation_rows: `30006`
- sealed_internal_test_rows: `30006`
- internal_split_strategy: `joint_treatment_outcome`
