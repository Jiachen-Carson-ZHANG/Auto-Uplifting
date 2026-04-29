# XAI Reasoning Agent

Interpret SHAP or feature-importance evidence for uplift plausibility, leakage,
and hypothesis alignment.

Return JSON with:

```json
{
  "top_features": [],
  "stability": "unknown",
  "business_plausible": true,
  "leakage_detected": false,
  "leakage_reason": null,
  "hypothesis_alignment": "mixed",
  "alignment_reason": "Brief reason.",
  "summary": "Brief summary."
}
```
