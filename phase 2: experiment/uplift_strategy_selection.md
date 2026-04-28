# Uplift Strategy Selection Agent

Selects the full uplift configuration for each trial — learner family, base estimator, feature recipe, split seed, and evaluation cutoff.

**Two-phase behavior:**
- **Warm-up** (fewer successful trials than learner families): cycles through all four families using default parameters to establish a baseline
- **Optimization** (sufficient history): picks the family with the highest mean AUUC from memory, then uses the LLM to reason over estimator and feature recipe choices

---

## Registries

### Learner Families

| Family | Class | Supported Estimators |
|--------|-------|----------------------|
| `SoloModel` | `sklift.models.SoloModel` | XGBoost, LightGBM, CatBoost |
| `TwoModels` | `sklift.models.TwoModels` | XGBoost, LightGBM, CatBoost |
| `ResponseModel` | `sklift.models.SoloModel` | XGBoost |
| `ClassTransformation` | `sklift.models.ClassTransformation` | XGBoost, LightGBM, CatBoost |

> `ResponseModel` is implemented as `SoloModel` with treatment as a feature.

### Estimator Defaults

| Estimator | Key Hyperparameters |
|-----------|---------------------|
| `XGBoost` | `n_estimators=100`, `max_depth=5`, `learning_rate=0.1`, `eval_metric="logloss"` |
| `LightGBM` | `n_estimators=100`, `max_depth=5`, `learning_rate=0.1`, `verbose=-1` |
| `CatBoost` | `iterations=100`, `depth=5`, `learning_rate=0.1`, `verbose=0` |
| `LogisticRegression` | `C=1.0`, `max_iter=1000` |

---

## System Prompt

```
You are an Uplift Strategy Selection Agent.
Given the knowledge base summary and a hypothesis, select the best uplift configuration.
Available learner families: SoloModel, TwoModels, ResponseModel, ClassTransformation
Available base estimators: XGBoost, LightGBM, CatBoost

Return ONLY valid JSON with keys:
  learner_family  (string)
  base_estimator  (string)
  feature_recipe  (string)
  split_seed      (integer)
  eval_cutoff     (float 0–1, proportion held out for evaluation)
  rationale       (1 sentence)
No markdown, no explanation — just the JSON object.
```

---

## Implementation

```python
WARMUP_ORDER = ["ResponseModel", "SoloModel", "TwoModels", "ClassTransformation"]

class UpliftStrategySelectionAgent:

    def __init__(self, memory: ExperimentMemory, llm: LLMClient):
        self.memory = memory
        self.llm = llm

    def run(
        self,
        hypothesis: HypothesisDecision,
        context: RetrievedContext,
    ) -> UpliftStrategy:

        records = self.memory.read_all()
        successful = [r for r in records if r.get("success")]

        # Warm-up: one trial per learner family using defaults
        if len(successful) < len(WARMUP_ORDER):
            used_families = {r["learner_family"] for r in successful}
            learner_family = next(
                (f for f in WARMUP_ORDER if f not in used_families),
                WARMUP_ORDER[0],
            )
            return UpliftStrategy(
                learner_family=learner_family,
                base_estimator="XGBoost",
                feature_recipe="rfm_baseline",
                split_seed=42,
                eval_cutoff=0.3,
                rationale=f"Warm-up trial for {learner_family} with default params.",
            )

        # Optimization: best mean AUUC per learner family
        family_auuc: dict[str, list[float]] = {}
        for r in successful:
            fam = r.get("learner_family", "SoloModel")
            auuc = r.get("metrics", {}).get("auuc", 0.0)
            family_auuc.setdefault(fam, []).append(auuc)

        mean_auuc = {f: sum(v) / len(v) for f, v in family_auuc.items()}
        best_family = max(mean_auuc, key=mean_auuc.get)

        user_msg = {
            "mean_auuc_by_family": mean_auuc,
            "best_family_so_far": best_family,
            "context_summary": context.summary,
            "active_hypothesis": hypothesis.hypothesis,
            "refuted_hypotheses": context.refuted_hypotheses,
        }

        raw = self.llm.chat(SYSTEM_PROMPT, json.dumps(user_msg, indent=2))
        parsed = _parse_json(raw)

        learner_family = parsed.get("learner_family", best_family)
        base_estimator = parsed.get("base_estimator", "XGBoost")

        # Validate against registry; fall back to safe defaults
        if learner_family not in LEARNER_REGISTRY:
            learner_family = best_family
        if base_estimator not in ESTIMATOR_DEFAULTS:
            base_estimator = "XGBoost"

        return UpliftStrategy(
            learner_family=learner_family,
            base_estimator=base_estimator,
            feature_recipe=parsed.get("feature_recipe", "rfm_baseline"),
            split_seed=int(parsed.get("split_seed", 42)),
            eval_cutoff=float(parsed.get("eval_cutoff", 0.3)),
            rationale=parsed.get("rationale", ""),
        )

    def get_estimator_params(self, base_estimator: str) -> dict:
        return dict(ESTIMATOR_DEFAULTS.get(base_estimator, {}))
```

---

## Notes

| Parameter | Type | Description |
|-----------|------|-------------|
| `memory` | `ExperimentMemory` | Source of all prior trial records |
| `llm` | `LLMClient` | LLM used to reason over estimator and recipe selection |
| `hypothesis` | `HypothesisDecision` | Active hypothesis from `HypothesisReasoningAgent` |
| `context` | `RetrievedContext` | Structured retrieval output from `CaseRetrievalAgent` |

**Safety:** LLM-returned `learner_family` and `base_estimator` are validated against the registry before use; invalid values fall back to the best-performing family and `XGBoost` respectively.
