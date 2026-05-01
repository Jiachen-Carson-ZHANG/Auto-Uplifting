# Best Autonomous Run — 2026-04-30

**Defensible champion:** class_transformation + LightGBM on `hybrid_safe_semantic_v1`
- Held-out raw Qini: **331.77** (`RUN-c5e6e86f`)
- Human notebook comparison: AutoLift raw held-out Qini **331.77** vs best `human_baseline_uplift.ipynb` held-out row **328.39** (+3.38 raw Qini). The human notebook remains higher on held-out uplift AUC and top-k lift rates.
- Val/HO gap: −1.69 raw Qini
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

Against the real human notebook comparison, defensible AutoLift is a narrow
held-out Qini leader: 331.77 vs 328.39 raw Qini. The human notebook remains ahead
on held-out uplift AUC and the top-k lift rates, so the performance claim should
be precise rather than exaggerated.

The agent's verdict ceiling correctly blocked two overfitting trials (inconclusive)
and selected the strongest supported main-run champion.

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
path that led to the final champion.

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

The validation-only tuned champion had validation raw Qini `357.375156` but
held-out audit Qini `317.725387`, so it does not replace `RUN-c5e6e86f`.
