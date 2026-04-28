# Hypothesis Reasoning Agent

Converts retrieved evidence and the current trial result into the next actionable hypothesis for the uplift modeling pipeline.

**Possible actions:**
- `validate` — current hypothesis is supported; worth further exploration
- `refute` — evidence contradicts the hypothesis; retire it
- `propose` — generate a new hypothesis based on gaps in the evidence

---

## System Prompt

```
You are a Hypothesis Reasoning Agent for an uplift modeling pipeline.
Given retrieved context and (optionally) the latest trial result, decide what to do next.
Choose one of three actions:
  validate  — the current hypothesis is supported and worth further exploration
  refute    — the evidence contradicts the current hypothesis; retire it
  propose   — generate a new hypothesis based on gaps in the evidence

Return ONLY valid JSON with keys:
  action       (string: "validate" | "refute" | "propose")
  hypothesis   (string: the active hypothesis going forward)
  evidence     (string: 1–2 sentences summarising supporting/contradicting evidence)
  confidence   (float 0–1)
No markdown, no explanation — just the JSON object.
```

---

## Implementation

```python
class HypothesisReasoningAgent:

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(
        self,
        context: RetrievedContext,
        current_hypothesis: Optional[str] = None,
        latest_trial: Optional[dict] = None,
    ) -> HypothesisDecision:

        user_msg = {
            "retrieved_context": {
                "supported_hypotheses": context.supported_hypotheses,
                "refuted_hypotheses": context.refuted_hypotheses,
                "best_learner_family": context.best_learner_family,
                "summary": context.summary,
            },
            "current_hypothesis": current_hypothesis,
            "latest_trial_result": latest_trial,
        }

        raw = self.llm.chat(SYSTEM_PROMPT, json.dumps(user_msg, indent=2))
        parsed = _parse_json(raw)

        return HypothesisDecision(
            action=parsed.get("action", "propose"),
            hypothesis=parsed.get("hypothesis", ""),
            evidence=parsed.get("evidence", ""),
            confidence=float(parsed.get("confidence", 0.5)),
        )
```

---

## Notes

| Parameter | Type | Description |
|-----------|------|-------------|
| `llm` | `LLMClient` | LLM used to reason over evidence and produce a decision |
| `context` | `RetrievedContext` | Structured retrieval output from `CaseRetrievalAgent` |
| `current_hypothesis` | `Optional[str]` | The hypothesis currently under evaluation (if any) |
| `latest_trial` | `Optional[dict]` | Most recent trial result to incorporate into reasoning |

**Default behaviour:** if the LLM response is missing fields, defaults to `action="propose"` and `confidence=0.5`.
