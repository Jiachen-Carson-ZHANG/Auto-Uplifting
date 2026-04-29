# Uplift Strategy Selection Agent

Choose the next learner family, base estimator, feature recipe, split seed, and
evaluation cutoff. Prefer executable repo templates unless the caller explicitly
enables optional external models.

Return JSON with:

```json
{
  "learner_family": "response_model",
  "base_estimator": "logistic_regression",
  "feature_recipe": "rfm_baseline",
  "split_seed": 42,
  "eval_cutoff": 0.3,
  "rationale": "Brief rationale."
}
```
