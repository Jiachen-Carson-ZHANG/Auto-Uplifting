# Best Autonomous Run — 2026-04-30

**Leakage-clean selected candidate:** two_model + LightGBM from validation-top-3 CV
- Selected run: `RUN-f1c30175` / `agentic_tune__UT-bc7585__p14`
- Selection rule: validation predictions choose top 3; 5-fold CV reranks those candidates on the original train+validation pool only; the internal test partition is sealed until final audit.
- CV mean normalized Qini: **0.396226**
- Sealed held-out raw Qini: **309.99**
- Human notebook comparison: final honest human CV champion is `solo_model_xgb` with CV mean normalized Qini **0.40949** and sealed test raw Qini **299.13**. AutoLift is slightly lower on CV mean normalized Qini but higher on sealed test raw/normalized Qini and top-k lift.

**Retrospective held-out best reference:** `RUN-c5e6e86f`
- Held-out raw Qini: **331.77**
- Status: useful audit/reference row, not the leakage-clean selected champion.
- XAI top features: age_clean, days_to_first_redeem, points_received_total_30d

## Files

| File | What it contains |
|---|---|
| `pipeline.log` | Full stage-by-stage run trace including `[plan]` (o3 hypothesis) and `[eval]` (o4-mini verdict) lines for every trial |
| `uplift_ledger.jsonl` | Append-only trial records: metrics, verdict, judge_narrative, xai_summary, policy_narrative, strategy_rationale |
| `final_report.md` | Auto-generated report: champion, benchmark, all trials table, policy recommendation, XAI explanation |
| `explainability/EXPLAINABILITY_REPORT.md` | Visual explanation pack: human-vs-AutoLift metric comparison, curves, deciles, XAI drivers, and agent reasoning timeline |
| `agentic_tuning_plan.json` | Quarantined original tuning plan; retained for audit, not final champion selection |
| `agentic_tuning_execution_summary.json` | Quarantined 32-trial tuning run summary; selector could see held-out metrics |
| `agentic_tuning_ledger.jsonl` | Combined ledger for the quarantined tuning specs |
| `agentic_tuning_plan_validation_only.json` | Patched validation-only tuning plan; no held-out metrics in candidate selection |
| `agentic_tuning_validation_only_execution_summary.json` | Patched tuning audit summary; validation-selected champion fails held-out audit |
| `agentic_tuning_validation_only_ledger.jsonl` | Combined ledger for the patched validation-only tuning specs |
| `validation_top3_cv_audit.md` | Leakage-clean top-3 CV audit; internal test partition excluded from CV |
| `validation_top3_cv_leaderboard.csv` | CV leaderboard for the top 3 validation-selected candidates |
| `honest_human_vs_autolift_cv_comparison.md` | Final honest comparison against the corrected human CV-selected champion |
| `honest_human_vs_autolift_cv_comparison.csv` | Machine-readable final comparison metrics |
| `tuned_submission_summary.json` | Quarantined tuned submission metadata; do not use for final claim |
| `tuned_xai_summary.json` | Quarantined tuned XAI summary for `RUN-2af274da` |
| `hypotheses.jsonl` | Hypothesis lifecycle records |

## Trial Summary

| Family | Estimator | Recipe | HO Norm Qini | Gap | Verdict |
|---|---|---|---:|---:|---|
| two_model | logistic_regression | rfm_baseline | 0.248 | +0.054 | baseline |
| class_transformation | gradient_boosting | rfm_baseline | 0.187 | −0.074 | inconclusive |
| two_model | xgboost | hybrid_safe_semantic_v1 | 0.249 | −0.124 | supported |
| class_transformation | xgboost | hybrid_safe_semantic_v1 | 0.313 | −0.049 | supported |
| two_model | lightgbm | hybrid_safe_semantic_v1 | 0.208 | −0.199 | inconclusive |
| **class_transformation** | **lightgbm** | **hybrid_safe_semantic_v1** | **0.337** | **−0.005** | **supported** |

## Key Finding

Switching from `rfm_baseline` (27k pre-issue transactions) to `hybrid_safe_semantic_v1`
(3.9M post-issue transactions with audited cutoff) shifted XAI top features from
pure demographics (`age_clean` only) to include behavioral signals
(`days_to_first_redeem`, `points_received_total_30d`).

Both sides are now compared under a leakage-clean CV selection boundary. The
human notebook selects `solo_model_xgb` by 5-fold CV after validation screening.
AutoLift selects `RUN-f1c30175` by validation-top-3 CV. Human is slightly ahead
on CV mean normalized Qini (`0.40949` vs `0.396226`), while AutoLift is ahead on
the sealed test normalized Qini (`0.248455` vs `0.20412`) and raw Qini
(`309.99` vs `299.13`).

The agent story should emphasize the end-to-end workflow: it generated trials,
tuned deterministically, detected the held-out leakage risk, reran a
validation-only top-3 CV audit, and preserved the reasoning and artifacts.

## Explainability Pack

Open `explainability/EXPLAINABILITY_REPORT.md` for report-ready visuals:

- `human_vs_autolift_topk.svg`
- `autolift_heldout_qini_curve.svg`
- `autolift_heldout_uplift_curve.svg`
- `autolift_decile_lift.svg`
- `autolift_xai_top_drivers.svg`
- `agent_reasoning_timeline.svg`

This supports the main agent story: AutoLift performs the full uplift workflow
end to end, records the rationale behind each trial, and exposes the decision
path. The current visual pack is for `RUN-c5e6e86f`, the retrospective held-out
best reference, not the strict validation+CV selected candidate `RUN-f1c30175`.

## Agentic Tuning Audit

`agentic_tuning_plan.json` was executed as a deterministic tuning stage, but a
post-run audit found that the original tuning selector could see internal
held-out metrics. That makes the tuned result useful for engineering diagnosis
but invalid for final champion claims.

Original quarantined tuned champion:

- Run ID: `RUN-2af274da`
- Held-out raw Qini: `338.601489`
- Status: quarantined, not report champion

A patched validation-only rerun selected candidates without held-out metrics:

- `two_model_xgboost`
- `two_model_lightgbm`

The validation-only raw-Qini winner had validation raw Qini `357.375156`.
A follow-up top-3 CV audit selected `RUN-f1c30175` by mean normalized CV Qini
without using the internal test partition. Its sealed held-out raw Qini is
`309.987113`, which is higher than the corrected human CV-selected champion's
sealed held-out raw Qini (`299.12559`). `RUN-c5e6e86f` remains only a
retrospective held-out-best reference.
