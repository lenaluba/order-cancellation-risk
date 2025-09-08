# Technical Anomaly Detection Report

## Executive Summary
- **Total anomalous transactions:** 11,285
- **Percentage of dataset:** 1.06%
- **Recommendation:** Review flagged transactions before model training

## 1. Quantity == ±1,000

**Business Impact:** Potential data entry errors or unusual bulk transactions

**Occurrences:** 44 (0.0041%)

**Sample transactions:**
|   Invoice | StockCode   | Description                      |   Quantity |   Price |   Customer ID |
|----------:|:------------|:---------------------------------|-----------:|--------:|--------------:|
|    501869 | 16014       | SMALL CHINESE STYLE SCISSOR      |       1000 |    0.32 |         13848 |
|    502267 | 21986       | PACK OF 12 PINK SPOT TISSUES     |       1000 |    0.5  |         17940 |
|    502267 | 21985       | PACK OF 12 HEARTS DESIGN TISSUES |       1000 |    0.5  |         17940 |
|    502267 | 21984       | PACK OF 12 PINK PAISLEY TISSUES  |       1000 |    0.5  |         17940 |
|    507450 | POST        | <NA>                             |       1000 |    0    |           nan |

## 2. Price > £10,000

**Business Impact:** Luxury items or potential pricing errors

**Occurrences:** 27 (0.0025%)

**Sample transactions:**
| Invoice   | StockCode   | Description   |   Quantity |   Price |   Customer ID |
|:----------|:------------|:--------------|-----------:|--------:|--------------:|
| C502262   | M           | Manual        |         -1 | 10953.5 |         12918 |
| 502263    | M           | Manual        |          1 | 10953.5 |         12918 |
| C502264   | M           | Manual        |         -1 | 10953.5 |         12918 |
| 502265    | M           | Manual        |          1 | 10953.5 |           nan |
| C512770   | M           | Manual        |         -1 | 25111.1 |         17399 |

## 3. Negative quantity without credit note

**Business Impact:** System error - sales should not have negative quantities

**Occurrences:** 3,457 (0.3239%)

**Sample transactions:**
|   Invoice | StockCode   | Description   |   Quantity |   Price |   Customer ID |
|----------:|:------------|:--------------|-----------:|--------:|--------------:|
|    489464 | 21733       | 85123a mixed  |        -96 |       0 |           nan |
|    489463 | 71477       | short         |       -240 |       0 |           nan |
|    489467 | 85123A      | 21733 mixed   |       -192 |       0 |           nan |
|    489521 | 21646       | <NA>          |        -50 |       0 |           nan |
|    489655 | 20683       | <NA>          |        -44 |       0 |           nan |

## 4. Zero price without credit note

**Business Impact:** Free items, promotions, or data quality issues

**Occurrences:** 6,202 (0.5811%)

**Sample transactions:**
|   Invoice | StockCode   | Description   |   Quantity |   Price |   Customer ID |
|----------:|:------------|:--------------|-----------:|--------:|--------------:|
|    489464 | 21733       | 85123a mixed  |        -96 |       0 |           nan |
|    489463 | 71477       | short         |       -240 |       0 |           nan |
|    489467 | 85123A      | 21733 mixed   |       -192 |       0 |           nan |
|    489521 | 21646       | <NA>          |        -50 |       0 |           nan |
|    489655 | 20683       | <NA>          |        -44 |       0 |           nan |

## 5. Description contains: TEST, MANUAL, ADJUST

**Business Impact:** Test data or manual adjustments - exclude from analysis

**Occurrences:** 1,555 (0.1457%)

**Sample transactions:**
| Invoice   | StockCode   | Description   |   Quantity |   Price |   Customer ID |
|:----------|:------------|:--------------|-----------:|--------:|--------------:|
| 489609    | M           | Manual        |          1 |    4    |           nan |
| C489651   | M           | Manual        |         -1 |    5.1  |         17804 |
| C489859   | M           | Manual        |         -1 |   69.57 |           nan |
| C490126   | M           | Manual        |         -1 |    5.95 |         15884 |
| C490129   | M           | Manual        |         -1 | 1998.49 |         15482 |

