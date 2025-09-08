# Business Rule Validation Report

## Customer Data Completeness
- **Customer ID missing:** 243,007 rows (22.77%)
  - *Impact:* Cannot perform customer lifetime value or retention analysis

## Transaction Integrity
- **Negative quantity (non-credits):** 3,457 rows (0.32%)
  - *Action:* Review for data entry errors or system issues
- **Zero/negative price (non-credits):** 6,207 rows (0.58%)
  - *Action:* Investigate promotional items or data quality issues

## Returns and Credits
- **Credit notes (Invoice prefix 'C'):** 19,494 rows (1.83%)
  - *Insight:* Return rate indicator for product quality analysis

## Data Duplication
- **Exact duplicate rows:** 12,133 rows (1.14%)
  - *Action:* Investigate potential double-posting in source systems

## Temporal Coverage
- **Date range:** 2009-12-01 to 2011-12-09
- **Duration:** 738 days

## Geographic Distribution
- **Total countries:** 43
- **Top 10 markets by transaction volume:**
  - United Kingdom: 981,330 transactions (91.94%)
  - EIRE: 17,866 transactions (1.67%)
  - Germany: 17,624 transactions (1.65%)
  - France: 14,330 transactions (1.34%)
  - Netherlands: 5,140 transactions (0.48%)
  - Spain: 3,811 transactions (0.36%)
  - Switzerland: 3,189 transactions (0.30%)
  - Belgium: 3,123 transactions (0.29%)
  - Portugal: 2,620 transactions (0.25%)
  - Australia: 1,913 transactions (0.18%)
