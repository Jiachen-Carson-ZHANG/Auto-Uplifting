# Ecommerce Feature Engineering Reference

## RFM Feature Family

RFM (Recency, Frequency, Monetary) is the foundational feature set for ecommerce lifecycle prediction.

### Recency
- **Definition:** Days since a customer's last transaction
- **Typical columns:** customer_id + order_date
- **Use cases:** Churn prediction (high recency = higher churn risk), repurchase timing
- **Pitfall:** Must use a cutoff date, not the observation date — otherwise future data leaks in

### Frequency
- **Definition:** Number of transactions in a defined window
- **Typical columns:** customer_id + order_date, windowed (30d, 90d, 365d)
- **Use cases:** Engagement scoring, customer segmentation
- **Pitfall:** Unbounded frequency (all-time count) includes future transactions if not windowed properly

### Monetary
- **Definition:** Total spend in a defined window
- **Typical columns:** customer_id + order_date + amount
- **Use cases:** Lifetime value prediction, high-value customer identification
- **Pitfall:** Include only completed transactions, not pending or refunded

## Temporal Features

### Days-since patterns
- Days since first purchase (customer tenure)
- Days since last purchase (recency)
- Days between purchases (inter-purchase interval)

### Seasonality
- Day of week of purchase
- Month of purchase
- Is-weekend indicator
- Holiday proximity

### Windowed aggregations
- Count, sum, mean, nunique in 7d / 30d / 90d / 365d windows
- Always require entity_key + time_col + explicit window

## Basket Features

### Average Order Value (AOV)
- Total spend / number of orders per customer
- Useful for segmentation and LTV

### Basket Size
- Average number of items per order
- Indicates engagement depth

### Category Diversity
- Number of unique product categories purchased
- Higher diversity often correlates with lower churn

## Ratio Features

### Cart-to-Purchase Rate
- purchases / cart_adds — measures conversion
- Must use safe_divide to handle zero denominators

### Discount Ratio
- discounted_orders / total_orders — measures price sensitivity
- High discount ratio may predict lower LTV

### Return Rate
- returned_orders / total_orders — measures satisfaction
- High return rate correlates with churn

## Common Pitfalls

1. **Target leakage from post-outcome data:** Features computed on data after the target event (e.g., using return data to predict purchase)
2. **Cross-customer leakage:** Aggregations that mix data across customers without proper entity_key grouping
3. **Unbounded windows:** All-time aggregations that include future data points
4. **Redundant features:** Multiple highly correlated RFM features that don't add information
5. **Missing value handling:** NaN in monetary features from customers with zero orders — use fillna(0), not dropna
