# Trial Spec Writer Agent

Write one readable planning trial spec for the selected uplift strategy.

Return JSON with:

```json
{
  "hypothesis": "RFM features improve treatment ranking.",
  "changes_from_previous": "Cold start first trial.",
  "expected_improvement": "Establish a baseline uplift ranking.",
  "model": "response_model + logistic_regression",
  "params": {"C": 1.0, "max_iter": 1000},
  "feature_recipe": "rfm_baseline",
  "stop_criteria": "Stop if Qini AUC does not improve after 3 trials."
}
```
