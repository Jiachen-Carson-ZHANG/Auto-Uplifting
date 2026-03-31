# Ecommerce Leakage Patterns Reference

## Pattern 1: Target Column in Feature Computation

**What:** The target column (e.g., `churned`, `repurchased`, `ltv`) appears directly in a feature formula.

**Example:** Computing `avg_spend_of_churned_customers` requires knowing who churned — the target.

**Detection:** Any feature spec that references the target column by name.

**Fix:** Remove the target column from all feature inputs.

## Pattern 2: Future-Looking Timestamps

**What:** Features use data from after the prediction time (cutoff date).

**Example:** Predicting whether a customer will churn in March, but using April purchase data to compute "recent_purchase_count."

**Detection:** Time-based features without an explicit cutoff or window boundary. Aggregations that use `max(date)` from the full dataset instead of a per-row cutoff.

**Fix:** All time-based features must specify a cutoff mode. Windowed templates enforce this by requiring `time_col` and `window_days`.

## Pattern 3: Post-Outcome Joins

**What:** Joining tables on events that happen after the target event.

**Example:** For churn prediction with a 90-day window, joining customer support tickets that were filed after the customer churned.

**Detection:** Join keys that include timestamps beyond the cutoff window.

**Fix:** Filter joined tables to only include events before the cutoff date.

## Pattern 4: Unbounded Aggregations

**What:** Aggregating data across all time without a window boundary.

**Example:** "Total lifetime purchases" computed over the entire history, including purchases after the prediction date.

**Detection:** Aggregation functions (count, sum, mean) without a `window_days` parameter.

**Fix:** Always specify an explicit window. The bounded DSL enforces this for time-requiring operations.

## Pattern 5: Missing Entity Boundaries

**What:** Aggregations that leak information across entities (customers).

**Example:** Computing "average order value" across all customers instead of per-customer, then using that global average as a feature.

**Detection:** Aggregations without an `entity_key` groupby.

**Fix:** All per-customer features must group by `entity_key`. The template system enforces this by requiring `entity_key` in all windowed templates.

## Quick Audit Checklist

- [ ] Target column not referenced in any feature computation
- [ ] All time-based features have explicit cutoff or window
- [ ] No joins on post-outcome events
- [ ] All aggregations are windowed (not all-time)
- [ ] All per-entity features group by entity_key
- [ ] No features derived from the target variable indirectly (e.g., target-encoded categories)
