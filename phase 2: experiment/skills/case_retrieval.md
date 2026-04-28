# Case Retrieval Agent

Reads prior trial history from experiment memory and surfaces structured context for downstream agents.

**Retrieves:**
- Similar feature recipes (with best AUUC)
- Prior supported / refuted hypotheses
- Best uplift learner family identified so far
- Failed runs to avoid repeating

---

## System Prompt

```
You are a Case Retrieval Agent in an agentic ML experimentation pipeline.
Your job: given a JSON list of prior trial records, extract structured retrieval context.
Return ONLY valid JSON with keys:
  similar_recipes       (list of dicts with recipe name + best auuc)
  supported_hypotheses  (list of hypothesis strings)
  refuted_hypotheses    (list of hypothesis strings)
  best_learner_family   (string)
  failed_runs           (list of dicts with trial_id + error_notes)
  summary               (1–2 sentence plain-English summary)
No markdown, no explanation — just the JSON object.
```

---

## Implementation

```python
class CaseRetrievalAgent:

    def __init__(self, memory: ExperimentMemory, llm: LLMClient):
        self.memory = memory
        self.llm = llm

    def run(self) -> RetrievedContext:
        records = self.memory.read_all()

        if not records:
            # Cold start — no prior history
            return RetrievedContext(
                similar_recipes=[],
                supported_hypotheses=[],
                refuted_hypotheses=[],
                best_learner_family="SoloModel",
                failed_runs=[],
                summary="Cold start — no prior trial history found.",
            )

        user_msg = (
            "Here are all prior trial records:\n"
            + json.dumps(records, indent=2)
            + "\n\nExtract the retrieval context."
        )

        raw = self.llm.chat(SYSTEM_PROMPT, user_msg)
        parsed = _parse_json(raw)

        return RetrievedContext(
            similar_recipes=parsed.get("similar_recipes", []),
            supported_hypotheses=parsed.get("supported_hypotheses", []),
            refuted_hypotheses=parsed.get("refuted_hypotheses", []),
            best_learner_family=parsed.get("best_learner_family", "SoloModel"),
            failed_runs=parsed.get("failed_runs", []),
            summary=parsed.get("summary", ""),
        )
```

---

## Notes

| Parameter | Type | Description |
|-----------|------|-------------|
| `memory` | `ExperimentMemory` | Source of prior trial records |
| `llm` | `LLMClient` | LLM used to extract structured context |

**Cold start behaviour:** if no prior records exist, returns safe defaults with `"SoloModel"` as the learner family.
