# Trial Spec Writer Agent

Produces a fully resolved, structured trial plan that downstream training agents can execute without further reasoning.

**Output (`TrialSpec`) contains:**
- Hypothesis being tested
- What changed from the previous run
- Expected metric improvement
- Exact model and hyperparameters
- Feature recipe
- Stop criteria

---

## System Prompt

```
You are a Trial Spec Writer Agent for an uplift modeling pipeline.
Given a hypothesis, an uplift strategy, and prior trial history, write a precise trial specification.

Return ONLY valid JSON with keys:
  trial_id              (string UUID)
  hypothesis            (string)
  changes_from_previous (string: what exactly changed from the last trial)
  expected_improvement  (string: e.g. "AUUC +0.01 over SoloModel baseline")
  model                 (string: e.g. "TwoModels + LightGBM")
  params                (dict of hyperparameters)
  feature_recipe        (string)
  stop_criteria         (string: condition under which this line of experiments should stop)
No markdown, no explanation — just the JSON object.
```

---

## Implementation

```python
class TrialSpecWriterAgent:

    def __init__(self, memory: ExperimentMemory, llm: LLMClient):
        self.memory = memory
        self.llm = llm

    def run(
        self,
        hypothesis: HypothesisDecision,
        strategy: UpliftStrategy,
        estimator_defaults: dict[str, Any],
    ) -> TrialSpec:

        last_trial = self.memory.best_trial("auuc")
        trial_id = str(uuid.uuid4())

        user_msg = {
            "trial_id": trial_id,
            "active_hypothesis": hypothesis.hypothesis,
            "hypothesis_action": hypothesis.action,
            "evidence": hypothesis.evidence,
            "strategy": asdict(strategy),
            "estimator_defaults": estimator_defaults,
            "last_best_trial": last_trial,
        }

        raw = self.llm.chat(SYSTEM_PROMPT, json.dumps(user_msg, indent=2))
        parsed = _parse_json(raw)

        return TrialSpec(
            trial_id=parsed.get("trial_id", trial_id),
            hypothesis=parsed.get("hypothesis", hypothesis.hypothesis),
            changes_from_previous=parsed.get("changes_from_previous", "N/A"),
            expected_improvement=parsed.get("expected_improvement", "N/A"),
            model=parsed.get("model", f"{strategy.learner_family} + {strategy.base_estimator}"),
            params=parsed.get("params", estimator_defaults),
            feature_recipe=parsed.get("feature_recipe", strategy.feature_recipe),
            stop_criteria=parsed.get("stop_criteria", "AUUC does not improve over 5 consecutive trials"),
        )
```

---

## Notes

| Parameter | Type | Description |
|-----------|------|-------------|
| `memory` | `ExperimentMemory` | Used to retrieve the current best trial by AUUC |
| `llm` | `LLMClient` | LLM used to author the structured trial specification |
| `hypothesis` | `HypothesisDecision` | Active hypothesis from `HypothesisReasoningAgent` |
| `strategy` | `UpliftStrategy` | Selected configuration from `UpliftStrategySelectionAgent` |
| `estimator_defaults` | `dict[str, Any]` | Default hyperparameters for the chosen base estimator |

**Default behaviour:** if LLM output is missing fields, fallbacks are applied — `trial_id` is preserved from the locally generated UUID, `model` is inferred from the strategy, and `stop_criteria` defaults to no improvement over 5 consecutive trials.
