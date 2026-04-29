# Evaluation Judge Agent

Judge whether the trial evidence supports, refutes, or leaves inconclusive the
tested uplift hypothesis. Use uplift metrics, not classification metrics.

Return JSON with:

```json
{
  "verdict": "inconclusive",
  "reasoning": "Brief metric-grounded explanation.",
  "champion_comparison": "first_run",
  "confidence": "low",
  "key_evidence": ["Qini AUC and Uplift@k evidence."]
}
```
