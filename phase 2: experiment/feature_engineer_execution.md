# Feature Engineering Execution Agent

Builds the feature table specified by the Trial Spec Writer Agent, guided by failed-recipe context from the Case Retrieval Agent, then validates the table and hands it to training.

**Responsibilities:**
- Builds the exact feature recipe described in `TrialSpec.feature_recipe`
- Validates exactly one row per `customer_id` (no duplicates, no missing customers)
- Checks for leakage — `target` and `treatment_flg` must not appear in the output table
- Sends the validated `FeatureTable` to the training agent

---

## Implementation

```python
import re
from dataclasses import dataclass
import pandas as pd


@dataclass
class FeatureTable:
    features_df : pd.DataFrame   # one row per customer_id, no target/treatment columns
    recipe_used : str
    trial_id    : str


class FeatureEngineeringExecutionAgent:
    """
    Builds the feature table requested by the Trial Spec Writer Agent,
    guided by what the Case Retrieval Agent flagged as failed recipes.

    Parameters
    ----------
    clients_df   : pd.DataFrame — X5 clients table
    purchases_df : pd.DataFrame — X5 purchases table
    train_df     : pd.DataFrame — uplift_train table (treatment_flg, target)
                   used only for leakage checking, never for feature building
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

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, spec: TrialSpec, context: RetrievedContext) -> FeatureTable:
        failed_recipes = {r["recipe"] for r in context.failed_runs if "recipe" in r}
        if spec.feature_recipe in failed_recipes:
            raise ValueError(
                f"Recipe '{spec.feature_recipe}' was flagged as failed in prior runs. "
                "Refusing to rebuild."
            )

        features_df = self._build_features(spec.feature_recipe)
        self._validate_grain(features_df)
        self._check_leakage(features_df)

        return FeatureTable(
            features_df=features_df,
            recipe_used=spec.feature_recipe,
            trial_id=spec.trial_id,
        )

    # ------------------------------------------------------------------
    # Feature builders
    # ------------------------------------------------------------------

    def _build_features(self, recipe: str) -> pd.DataFrame:
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
            # unrecognised tokens are silently skipped

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

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_grain(self, df: pd.DataFrame) -> None:
        n_rows      = len(df)
        n_customers = df["customer_id"].nunique()
        if n_rows != n_customers:
            dupes = df[df.duplicated("customer_id", keep=False)]["customer_id"].unique()
            raise ValueError(
                f"Grain check failed: {n_rows} rows but only {n_customers} unique customer_ids. "
                f"Example duplicates: {list(dupes[:5])}"
            )

    def _check_leakage(self, df: pd.DataFrame) -> None:
        direct_leak = self.LEAKAGE_COLS & set(df.columns)
        if direct_leak:
            raise ValueError(f"Leakage check failed: forbidden columns {direct_leak} present.")

        train_only    = set(self.train_df.columns) - {"customer_id", "client_id"}
        indirect_leak = (train_only & set(df.columns)) - {"customer_id"}
        if indirect_leak:
            raise ValueError(
                f"Leakage check failed: train-table columns {indirect_leak} found in feature table."
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prepare_purchases(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["transaction_datetime"] = pd.to_datetime(df["transaction_datetime"])
        return df.sort_values("transaction_datetime")
```

---

## Notes

| Parameter | Type | Description |
|-----------|------|-------------|
| `clients_df` | `pd.DataFrame` | X5 clients table — source of demographic features |
| `purchases_df` | `pd.DataFrame` | X5 purchases table — source of behavioural features |
| `train_df` | `pd.DataFrame` | Uplift train table — used **only** for leakage detection, never for feature building |
| `spec` | `TrialSpec` | Specifies `feature_recipe` (string) and `trial_id` |
| `context` | `RetrievedContext` | Failed-recipe list from `CaseRetrievalAgent`; recipes in `failed_runs` are refused |

**Grain validation:** raises `ValueError` if any `customer_id` appears more than once; reports the first five duplicate IDs so the caller can diagnose the offending builder.

**Leakage check (two layers):**
1. *Direct* — `target` and `treatment_flg` must not appear in the feature table.
2. *Indirect* — any column exclusive to `train_df` (beyond the join key) must not appear in the feature table.

**Recipe dispatch:** `feature_recipe` is split on `+` and each token is matched to a builder method. Unrecognised tokens are silently skipped. All builders left-join onto the base `customer_id` spine; missing values are filled with `0`.

**Supported recipe tokens:**

| Token pattern | Builder | Output columns |
|---|---|---|
| `rfm` | `_rfm_features()` | `recency_days`, `frequency`, `monetary_total` |
| `demographic` | `_demographic_features()` | `age`, `gender_*` dummies |
| `purchase_frequency_Nd` | `_purchase_frequency_features(N)` | `freq_Nd` |
| `basket` | `_basket_features()` | `avg_basket_size` |
 