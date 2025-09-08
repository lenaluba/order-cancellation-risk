# Basket Value Analysis & Outlier Treatment Strategy

## Distribution Statistics
- **Total baskets analyzed:** 41,944
- **Mean basket value:** £496.24
- **Median basket value:** £295.69
- **Standard deviation:** £1540.13
- **Coefficient of variation:** 3.10

## Percentile Analysis
- **99th percentile:** £4,403.89
- **99.5th percentile:** £6,316.67
- **99.9th percentile:** £16,483.29

## Extreme Values
**Top 5 basket values:**
1. £168,469.60
2. £77,183.60
3. £52,940.94
4. £50,653.91
5. £49,844.99

## Outlier Treatment Recommendations

### Conservative Approach (99th percentile): £4,403.89
- **Transactions affected:** 420 (1.00%)
- **Business rationale:** Removes only extreme outliers, preserves high-value customer data

### Balanced (Recommended) Approach (99.5th percentile): £6,316.67
- **Transactions affected:** 210 (0.50%)
- **Business rationale:** Optimal trade-off between data integrity and model stability

### Liberal Approach (99.9th percentile): £16,483.29
- **Transactions affected:** 42 (0.10%)
- **Business rationale:** Minimal intervention, retains nearly all transactions

## Implementation Strategy
1. **For predictive modeling:** Cap at 99.5th percentile (£6316.67)
2. **For descriptive analytics:** Flag but retain all values
3. **For financial reporting:** No capping, use actual values
4. **Next steps:** Review top baskets for business validity before finalizing
