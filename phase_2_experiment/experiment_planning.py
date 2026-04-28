"""
experiment_planning.py
======================
Phase II — Experiment Planning Agents for BT5153 Group 1
Agents: CaseRetrievalAgent, HypothesisReasoningAgent,
        UpliftStrategySelectionAgent, TrialSpecWriterAgent

LLM backend is abstracted via LLMClient — swap in any provider later.
Memory layer: JSONL file (same schema as pipeline.py knowledge base).
Uplift library: scikit-uplift (sklift)
"""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Skill loader — reads system prompts from Skills/{name}.md
# ---------------------------------------------------------------------------

def load_skill_prompt(skill_name: str) -> str:
    """Extract the ## System Prompt code block from Skills/{skill_name}.md."""
    path = Path(__file__).parent / "Skills" / f"{skill_name}.md"
    text = path.read_text(encoding="utf-8")
    match = re.search(r"## System Prompt\s*\n+```[^\n]*\n(.*?)```", text, re.DOTALL)
    if not match:
        raise ValueError(f"No '## System Prompt' code block found in {path}")
    return match.group(1).strip()


# ---------------------------------------------------------------------------
# LLM abstraction — swap backend by changing LLMClient
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Thin wrapper around any chat-completion API.
    Default: stub that returns the system prompt echo so the pipeline
    runs without a live key.  Replace `_call` with your provider SDK.

    To use Anthropic:
        client = LLMClient(provider="anthropic", api_key=os.environ["ANTHROPIC_API_KEY"])
    To use OpenAI:
        client = LLMClient(provider="openai", api_key=os.environ["OPENAI_API_KEY"])
    """

    def __init__(
        self,
        provider: str = "stub",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.provider = provider
        self.api_key = api_key
        self.model = model or self._default_model()

    def _default_model(self) -> str:
        defaults = {
            "anthropic": "claude-sonnet-4-20250514",
            "openai": "gpt-4o",
            "stub": "stub",
        }
        return defaults.get(self.provider, "stub")

    def chat(self, system: str, user: str) -> str:
        """Return the assistant reply as a plain string."""
        if self.provider == "anthropic":
            return self._call_anthropic(system, user)
        if self.provider == "openai":
            return self._call_openai(system, user)
        # Stub: return a minimal JSON so downstream parsing doesn't crash
        return self._stub_reply(system, user)

    # ------------------------------------------------------------------
    # Provider implementations
    # ------------------------------------------------------------------

    def _call_openai(self, system: str, user: str) -> str:
        from openai import OpenAI  # pip install openai
        client = OpenAI(api_key=self.api_key)
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content

    def _stub_reply(self, system: str, user: str) -> str:
        """
        Deterministic stub so the full pipeline runs offline.
        Returns valid JSON matching the expected schema for each agent.
        Detected by keywords in the system prompt.
        """
        if "case retrieval" in system.lower():
            return json.dumps({
                "similar_recipes": [],
                "supported_hypotheses": [],
                "refuted_hypotheses": [],
                "best_learner_family": "SoloModel",
                "failed_runs": [],
                "summary": "No prior trials found — cold start."
            })
        if "hypothesis" in system.lower():
            return json.dumps({
                "action": "propose",
                "hypothesis": "RFM features with 90-day recency window improve AUUC",
                "evidence": "Cold start — no prior evidence",
                "confidence": 0.5
            })
        if "uplift strategy" in system.lower():
            return json.dumps({
                "learner_family": "SoloModel",
                "base_estimator": "XGBoost",
                "feature_recipe": "rfm_baseline",
                "split_seed": 42,
                "eval_cutoff": 0.3,
                "rationale": "Default warm-start selection"
            })
        if "trial spec" in system.lower():
            return json.dumps({
                "trial_id": str(uuid.uuid4()),
                "hypothesis": "RFM features with 90-day recency window improve AUUC",
                "changes_from_previous": "Cold start — first trial",
                "expected_improvement": "Establish AUUC baseline",
                "model": "SoloModel + XGBoost",
                "params": {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1},
                "feature_recipe": "rfm_baseline",
                "stop_criteria": "AUUC < 0.50 after 5 consecutive trials"
            })
        return "{}"


# ---------------------------------------------------------------------------
# Shared data structures
# ---------------------------------------------------------------------------

@dataclass
class TrialRecord:
    """One row in the JSONL knowledge base."""
    trial_id: str
    timestamp: str
    learner_family: str          # SoloModel | TwoModels | ResponseModel | ClassTransformation
    base_estimator: str          # XGBoost | LightGBM | CatBoost | LogisticRegression
    feature_recipe: str
    hyperparams: dict[str, Any]
    metrics: dict[str, float]    # auuc, roc_auc, f1, accuracy, train_time
    hypothesis: str
    hypothesis_status: str       # supported | refuted | inconclusive
    success: bool
    error_notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RetrievedContext:
    """Output of CaseRetrievalAgent."""
    similar_recipes: list[dict]
    supported_hypotheses: list[str]
    refuted_hypotheses: list[str]
    best_learner_family: str
    failed_runs: list[dict]
    summary: str


@dataclass
class HypothesisDecision:
    """Output of HypothesisReasoningAgent."""
    action: str           # "validate" | "refute" | "propose"
    hypothesis: str       # the active hypothesis going forward
    evidence: str
    confidence: float     # 0–1


@dataclass
class UpliftStrategy:
    """Output of UpliftStrategySelectionAgent."""
    learner_family: str
    base_estimator: str
    feature_recipe: str
    split_seed: int
    eval_cutoff: float
    rationale: str


@dataclass
class TrialSpec:
    """Output of TrialSpecWriterAgent — the fully resolved trial plan."""
    trial_id: str
    hypothesis: str
    changes_from_previous: str
    expected_improvement: str
    model: str
    params: dict[str, Any]
    feature_recipe: str
    stop_criteria: str


@dataclass
class FeatureTable:
    """Output of FeatureEngineeringExecutionAgent."""
    feature_df: pd.DataFrame
    recipe_name: str
    feature_names: list
    n_customers: int
    validation_passed: bool
    leakage_check_passed: bool
    validation_notes: list
    skipped_families: list


# ---------------------------------------------------------------------------
# Experiment Memory (JSONL)
# ---------------------------------------------------------------------------

class ExperimentMemory:
    """
    Append-only JSONL knowledge base.
    Compatible with the KnowledgeBase schema used in pipeline.py.
    """

    def __init__(self, path: str = "logs/knowledge_base.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: TrialRecord) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(record.to_dict()) + "\n")

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        records = []
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return records

    def best_trial(self, metric: str = "auuc") -> Optional[dict]:
        records = [r for r in self.read_all() if r.get("success")]
        if not records:
            return None
        return max(records, key=lambda r: r.get("metrics", {}).get(metric, 0.0))

    def summary_stats(self) -> dict:
        records = self.read_all()
        successful = [r for r in records if r.get("success")]
        if not successful:
            return {"total": len(records), "successful": 0, "best_auuc": None}
        best = self.best_trial("auuc")
        return {
            "total": len(records),
            "successful": len(successful),
            "best_auuc": best["metrics"].get("auuc") if best else None,
            "best_trial_id": best["trial_id"] if best else None,
        }


# ---------------------------------------------------------------------------
# Agent 1: CaseRetrievalAgent
# ---------------------------------------------------------------------------

class CaseRetrievalAgent:
    """
    Reads prior trial history from experiment memory and surfaces:
      - similar feature recipes
      - prior supported / refuted hypotheses
      - best uplift learner family so far
      - failed runs to avoid repeating
    """

    SYSTEM_PROMPT = load_skill_prompt("case_retrieval")

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

        raw = self.llm.chat(self.SYSTEM_PROMPT, user_msg)
        parsed = _parse_json(raw)

        return RetrievedContext(
            similar_recipes=parsed.get("similar_recipes", []),
            supported_hypotheses=parsed.get("supported_hypotheses", []),
            refuted_hypotheses=parsed.get("refuted_hypotheses", []),
            best_learner_family=parsed.get("best_learner_family", "SoloModel"),
            failed_runs=parsed.get("failed_runs", []),
            summary=parsed.get("summary", ""),
        )


# ---------------------------------------------------------------------------
# Agent 2: HypothesisReasoningAgent
# ---------------------------------------------------------------------------

class HypothesisReasoningAgent:
    """
    Converts retrieved evidence + current results into the next trial idea.
    Actions: validate | refute | propose
    """

    SYSTEM_PROMPT = load_skill_prompt("hypothesis_reasoning")

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

        raw = self.llm.chat(self.SYSTEM_PROMPT, json.dumps(user_msg, indent=2))
        parsed = _parse_json(raw)

        return HypothesisDecision(
            action=parsed.get("action", "propose"),
            hypothesis=parsed.get("hypothesis", ""),
            evidence=parsed.get("evidence", ""),
            confidence=float(parsed.get("confidence", 0.5)),
        )


# ---------------------------------------------------------------------------
# Agent 3: UpliftStrategySelectionAgent
# ---------------------------------------------------------------------------

# Supported learner families and base estimators (scikit-uplift)
LEARNER_REGISTRY = {
    "SoloModel": {
        "class": "sklift.models.SoloModel",
        "estimators": ["XGBoost", "LightGBM", "CatBoost"],
    },
    "TwoModels": {
        "class": "sklift.models.TwoModels",
        "estimators": ["XGBoost", "LightGBM", "CatBoost"],
    },
    "ResponseModel": {
        "class": "sklift.models.SoloModel",   # response-model is SoloModel with treatment as feature
        "estimators": ["XGBoost"],
    },
    "ClassTransformation": {
        "class": "sklift.models.ClassTransformation",
        "estimators": ["XGBoost", "LightGBM", "CatBoost"],
    },
}

ESTIMATOR_DEFAULTS = {
    "XGBoost":          {"n_estimators": 100, "max_depth": 5,  "learning_rate": 0.1, "use_label_encoder": False, "eval_metric": "logloss"},
    "LightGBM":         {"n_estimators": 100, "max_depth": 5,  "learning_rate": 0.1, "verbose": -1},
    "CatBoost":         {"iterations": 100,   "depth": 5,      "learning_rate": 0.1, "verbose": 0},
    "LogisticRegression": {"C": 1.0, "max_iter": 1000},
}


class UpliftStrategySelectionAgent:
    """
    Selects:
      - uplift learner family (SoloModel | TwoModels | ResponseModel | ClassTransformation)
      - base estimator (XGBoost | LightGBM | CatBoost)
      - feature recipe
      - split seed
      - evaluation cutoff

    During warm-up (no prior trials): cycles through all four learner families.
    During optimization: picks learner family with highest mean AUUC from memory.
    LLM is used to add reasoning and rationale.
    """

    SYSTEM_PROMPT = load_skill_prompt("uplift_strategy_selection")

    WARMUP_ORDER = ["ResponseModel", "SoloModel", "TwoModels", "ClassTransformation"]

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

        # ---- Warm-up: one trial per learner family using defaults ----
        if len(successful) < len(self.WARMUP_ORDER):
            used_families = {r["learner_family"] for r in successful}
            learner_family = next(
                (f for f in self.WARMUP_ORDER if f not in used_families),
                self.WARMUP_ORDER[0],
            )
            base_estimator = "XGBoost"
            feature_recipe = "rfm_baseline"
            rationale = f"Warm-up trial for {learner_family} with default params."
            return UpliftStrategy(
                learner_family=learner_family,
                base_estimator=base_estimator,
                feature_recipe=feature_recipe,
                split_seed=42,
                eval_cutoff=0.3,
                rationale=rationale,
            )

        # ---- Optimization: best mean AUUC per learner family ----
        family_auuc: dict[str, list[float]] = {}
        for r in successful:
            fam = r.get("learner_family", "SoloModel")
            auuc = r.get("metrics", {}).get("auuc", 0.0)
            family_auuc.setdefault(fam, []).append(auuc)

        mean_auuc = {f: sum(v) / len(v) for f, v in family_auuc.items()}
        best_family = max(mean_auuc, key=mean_auuc.get)

        # Let LLM reason over the choice and pick estimator + recipe
        user_msg = {
            "mean_auuc_by_family": mean_auuc,
            "best_family_so_far": best_family,
            "context_summary": context.summary,
            "active_hypothesis": hypothesis.hypothesis,
            "refuted_hypotheses": context.refuted_hypotheses,
        }

        raw = self.llm.chat(self.SYSTEM_PROMPT, json.dumps(user_msg, indent=2))
        parsed = _parse_json(raw)

        learner_family = parsed.get("learner_family", best_family)
        base_estimator = parsed.get("base_estimator", "XGBoost")

        # Safety: validate against registry
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
        """Return default hyperparameter dict for a given base estimator."""
        return dict(ESTIMATOR_DEFAULTS.get(base_estimator, {}))


# ---------------------------------------------------------------------------
# Agent 4: TrialSpecWriterAgent
# ---------------------------------------------------------------------------

class TrialSpecWriterAgent:
    """
    Produces a fully resolved, structured trial plan that downstream
    training agents can execute without further reasoning.

    Output (TrialSpec) contains:
      - hypothesis being tested
      - what changed from previous run
      - expected metric improvement
      - exact model + params
      - feature recipe
      - stop criteria
    """

    SYSTEM_PROMPT = load_skill_prompt("trial_spec_writer")

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

        raw = self.llm.chat(self.SYSTEM_PROMPT, json.dumps(user_msg, indent=2))
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


# ---------------------------------------------------------------------------
# Agent 5: FeatureEngineeringExecutionAgent
# ---------------------------------------------------------------------------

class FeatureEngineeringExecutionAgent:
    """
    Builds the feature table specified by the TrialSpec and validates it.
    Returns a FeatureTable with boolean validation flags (no exceptions raised)
    so Cell 8's assertion gate controls the hard stop.
    """

    LEAKAGE_COLS = {"target", "treatment_flg"}

    def __init__(
        self,
        clients_df: pd.DataFrame,
        purchases_df: pd.DataFrame,
        train_df: pd.DataFrame,
    ):
        self.clients_df   = clients_df.copy()
        self.purchases_df = self._prepare_purchases(purchases_df)
        self.train_df     = train_df.copy()

    def run(self, trial_spec: TrialSpec, context: RetrievedContext) -> FeatureTable:
        failed_recipes = {r["recipe"] for r in context.failed_runs if "recipe" in r}
        if trial_spec.feature_recipe in failed_recipes:
            raise ValueError(
                f"Recipe '{trial_spec.feature_recipe}' was flagged as failed in prior runs."
            )

        skipped: list = []
        feature_df = self._build_features(trial_spec.feature_recipe, skipped)

        validation_passed    = True
        leakage_check_passed = True
        validation_notes: list = []

        n_rows      = len(feature_df)
        n_customers = feature_df["customer_id"].nunique()
        if n_rows != n_customers:
            validation_passed = False
            dupes = feature_df[feature_df.duplicated("customer_id", keep=False)]["customer_id"].unique()
            validation_notes.append(
                f"Grain check failed: {n_rows} rows, {n_customers} unique customer_ids. "
                f"Example duplicates: {list(dupes[:5])}"
            )

        direct_leak = self.LEAKAGE_COLS & set(feature_df.columns)
        if direct_leak:
            leakage_check_passed = False
            validation_notes.append(f"Direct leakage: forbidden columns {direct_leak} present.")

        train_only    = set(self.train_df.columns) - {"customer_id", "client_id"}
        indirect_leak = (train_only & set(feature_df.columns)) - {"customer_id"}
        if indirect_leak:
            leakage_check_passed = False
            validation_notes.append(f"Indirect leakage: train-only columns {indirect_leak} found.")

        if validation_passed and leakage_check_passed:
            validation_notes.append("All checks passed.")

        feature_cols = [c for c in feature_df.columns if c != "customer_id"]

        return FeatureTable(
            feature_df=feature_df,
            recipe_name=trial_spec.feature_recipe,
            feature_names=feature_cols,
            n_customers=n_customers,
            validation_passed=validation_passed,
            leakage_check_passed=leakage_check_passed,
            validation_notes=validation_notes,
            skipped_families=skipped,
        )

    def _build_features(self, recipe: str, skipped: list) -> pd.DataFrame:
        base   = self.clients_df[["client_id"]].rename(columns={"client_id": "customer_id"})
        tokens = [t.strip().lower() for t in recipe.split("+")]
        frames = []

        for token in tokens:
            if "rfm" in token:
                frames.append(self._rfm_features())
            elif "demographic" in token:
                frames.append(self._demographic_features())
            elif "purchase_frequency" in token:
                match = re.search(r"(\d+)d", token)
                days  = int(match.group(1)) if match else 90
                frames.append(self._purchase_frequency_features(days))
            elif "basket" in token:
                frames.append(self._basket_features())
            else:
                skipped.append(token)

        result = base.copy()
        for frame in frames:
            result = result.merge(frame, on="customer_id", how="left")
        return result.fillna(0)

    def _rfm_features(self) -> pd.DataFrame:
        ref_date = self.purchases_df["transaction_datetime"].max()
        return (
            self.purchases_df
            .groupby("client_id")
            .agg(
                recency_days   =("transaction_datetime", lambda x: (ref_date - x.max()).days),
                frequency      =("transaction_datetime", "count"),
                monetary_total =("purchase_sum", "sum"),
            )
            .reset_index()
            .rename(columns={"client_id": "customer_id"})
        )

    def _demographic_features(self) -> pd.DataFrame:
        want      = ["client_id", "age", "gender_cd"]
        available = [c for c in want if c in self.clients_df.columns]
        df        = self.clients_df[available].rename(columns={"client_id": "customer_id"})
        if "gender_cd" in df.columns:
            df = pd.get_dummies(df, columns=["gender_cd"], prefix="gender", drop_first=False)
        return df

    def _purchase_frequency_features(self, days: int) -> pd.DataFrame:
        ref_date = self.purchases_df["transaction_datetime"].max()
        cutoff   = ref_date - pd.Timedelta(days=days)
        window   = self.purchases_df[self.purchases_df["transaction_datetime"] >= cutoff]
        return (
            window
            .groupby("client_id")
            .agg(**{f"freq_{days}d": ("transaction_datetime", "count")})
            .reset_index()
            .rename(columns={"client_id": "customer_id"})
        )

    def _basket_features(self) -> pd.DataFrame:
        return (
            self.purchases_df
            .groupby("client_id")
            .agg(avg_basket_size=("purchase_sum", "mean"))
            .reset_index()
            .rename(columns={"client_id": "customer_id"})
        )

    def _prepare_purchases(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["transaction_datetime"] = pd.to_datetime(df["transaction_datetime"])
        return df.sort_values("transaction_datetime")


# ---------------------------------------------------------------------------
# Orchestrator: ExperimentPlanningPhase
# ---------------------------------------------------------------------------

class ExperimentPlanningPhase:
    """
    Runs all four Experiment Planning agents in sequence and returns
    a TrialSpec ready for execution by the training agents.

    Usage
    -----
    planner = ExperimentPlanningPhase(
        memory=ExperimentMemory("logs/knowledge_base.jsonl"),
        llm=LLMClient(provider="anthropic", api_key="sk-..."),
    )
    spec = planner.run(current_hypothesis="RFM 90-day recency improves AUUC")
    print(spec)
    """

    def __init__(self, memory: ExperimentMemory, llm: LLMClient):
        self.memory = memory
        self.llm = llm
        self.case_retrieval = CaseRetrievalAgent(memory, llm)
        self.hypothesis_reasoning = HypothesisReasoningAgent(llm)
        self.strategy_selection = UpliftStrategySelectionAgent(memory, llm)
        self.trial_spec_writer = TrialSpecWriterAgent(memory, llm)

    def run(self, current_hypothesis: Optional[str] = None) -> TrialSpec:
        print("[ExperimentPlanning] Step 1/4 — Case Retrieval")
        context = self.case_retrieval.run()
        print(f"  → {context.summary}")

        print("[ExperimentPlanning] Step 2/4 — Hypothesis Reasoning")
        hypothesis = self.hypothesis_reasoning.run(
            context=context,
            current_hypothesis=current_hypothesis,
            latest_trial=self.memory.best_trial("auuc"),
        )
        print(f"  → [{hypothesis.action.upper()}] {hypothesis.hypothesis}")

        print("[ExperimentPlanning] Step 3/4 — Uplift Strategy Selection")
        strategy = self.strategy_selection.run(hypothesis, context)
        print(f"  → {strategy.learner_family} + {strategy.base_estimator} | recipe: {strategy.feature_recipe}")

        print("[ExperimentPlanning] Step 4/4 — Trial Spec Writer")
        estimator_defaults = self.strategy_selection.get_estimator_params(strategy.base_estimator)
        spec = self.trial_spec_writer.run(hypothesis, strategy, estimator_defaults)
        print(f"  → Trial {spec.trial_id[:8]}… | {spec.model}")
        print(f"     Hypothesis : {spec.hypothesis}")
        print(f"     Changes    : {spec.changes_from_previous}")
        print(f"     Stop when  : {spec.stop_criteria}")

        return spec


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _parse_json(text: str) -> dict:
    """
    Safely parse LLM output that may be wrapped in ```json ... ``` fences.
    Falls back to an empty dict on failure.
    """
    text = text.strip()
    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Attempt to extract first {...} block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {}


# ---------------------------------------------------------------------------
# Quick smoke test (run as script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        memory = ExperimentMemory(os.path.join(tmp, "kb.jsonl"))
        llm = LLMClient(provider="stub")  # no API key needed for smoke test

        planner = ExperimentPlanningPhase(memory=memory, llm=llm)

        print("=" * 60)
        print("SMOKE TEST — cold start (no prior trials)")
        print("=" * 60)
        spec = planner.run()
        print("\nFinal TrialSpec:")
        print(json.dumps(asdict(spec), indent=2))

        # Simulate one completed trial being logged
        record = TrialRecord(
            trial_id=spec.trial_id,
            timestamp=datetime.utcnow().isoformat(),
            learner_family="ResponseModel",
            base_estimator="XGBoost",
            feature_recipe="rfm_baseline",
            hyperparams=spec.params,
            metrics={"auuc": 0.541, "roc_auc": 0.623, "f1": 0.48, "accuracy": 0.71, "train_time": 12.3},
            hypothesis=spec.hypothesis,
            hypothesis_status="supported",
            success=True,
        )
        memory.append(record)

        print("\n" + "=" * 60)
        print("SMOKE TEST — after 1 completed trial")
        print("=" * 60)
        spec2 = planner.run(current_hypothesis=spec.hypothesis)
        print("\nFinal TrialSpec:")
        print(json.dumps(asdict(spec2), indent=2))
