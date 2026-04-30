# Best Autonomous Run — 2026-04-30

**Champion after deterministic agentic tuning:** class_transformation + LightGBM on `hybrid_safe_semantic_v1`
- Held-out raw Qini: **338.60** (`RUN-2af274da`)
- Human notebook comparison: tuned AutoLift raw held-out Qini **338.60** vs best `human_baseline_uplift.ipynb` held-out row **328.39** (+10.21 raw Qini). Tuned AutoLift is also higher on held-out uplift AUC and uplift@5/@20, while the human notebook remains higher on uplift@10 and uplift@30.
- Val/HO gap: −17.28 raw Qini (larger than the original champion, but still the strongest stability-adjusted tuning result)
- XAI top features: age_clean, days_to_first_redeem, points_received_total_30d

## Files

| File | What it contains |
|---|---|
| `pipeline.log` | Full stage-by-stage run trace including `[plan]` (o3 hypothesis) and `[eval]` (o4-mini verdict) lines for every trial |
| `uplift_ledger.jsonl` | Append-only trial records: metrics, verdict, judge_narrative, xai_summary, policy_narrative, strategy_rationale |
| `final_report.md` | Auto-generated report: champion, benchmark, all trials table, policy recommendation, XAI explanation |
| `explainability/EXPLAINABILITY_REPORT.md` | Visual explanation pack: human-vs-AutoLift metric comparison, curves, deciles, XAI drivers, and agent reasoning timeline |
| `agentic_tuning_plan.json` | Deterministic tuning plan for the top two internal AutoLift candidates; executed in `agentic_tuning_execution_summary.json` |
| `agentic_tuning_execution_summary.json` | Completed 32-trial tuning run summary and tuned champion metadata |
| `agentic_tuning_ledger.jsonl` | Combined ledger for all trained agentic tuning specs |
| `tuned_submission_summary.json` | Validated tuned champion submission metadata; CSV remains in `artifacts/uplift/run_20260430_221602/agentic_tuning/uplift_submission.csv` |
| `tuned_xai_summary.json` | Model-agnostic permutation explanation for tuned champion `RUN-2af274da` |
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

Against the real human notebook comparison, tuned AutoLift is the held-out Qini
leader: 338.60 vs 328.39 raw Qini. It is also ahead on held-out uplift AUC,
uplift@5%, and uplift@20%; the human notebook remains ahead on uplift@10% and
uplift@30%, so the claim should still be precise rather than exaggerated.

The agent's verdict ceiling correctly blocked two overfitting trials (inconclusive)
and selected the most stable champion via stability-adjusted scoring.

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

## Agentic Tuning Plan

`agentic_tuning_plan.json` was executed as a deterministic tuning stage. It
selects the top two candidates only from the AutoLift runtime ledger, asks the
LLM for bounded search rooms, validates those rooms against programmatic
guardrails, and samples 32 total trial specs with tuning seed `20260501`.

Selected candidates:

- `class_transformation_lightgbm` from `RUN-c5e6e86f`
- `class_transformation_xgboost` from `RUN-f7bdb1dc`

Tuned champion:

- Run ID: `RUN-2af274da`
- Spec: `AT-01-02-class-transformation-lightgbm`
- Params: `learning_rate=0.03`, `max_depth=3`, `n_estimators=300`, `num_leaves=31`
- Validation raw Qini: `355.881014`
- Held-out raw Qini: `338.601489`
- Held-out uplift AUC: `0.06397`
