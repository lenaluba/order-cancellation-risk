# Statistical Distribution Analysis

## Shapiro-Wilk Normality Test Results

| Variable | P-Value | Statistic | Normal? | Sample Size | Notes |
|----------|---------|-----------|---------|-------------|-------|
| Quantity | 0.000000 | 0.166783 | No | 5000 |  |
| Unit Price | 0.000000 | 0.069200 | No | 5000 |  |
| Basket Value | 0.000000 | 0.516407 | No | 5000 |  |

## Interpretation Guide
- **P-value < 0.05**: Reject normality assumption (consider log transformation)
- **Large samples**: Test becomes oversensitive; visual inspection recommended
- **Business Impact**: Non-normal distributions may require:
  - Log or Box-Cox transformation for linear models
  - Robust scaling for outlier-sensitive algorithms
  - Tree-based models which handle non-normality naturally
