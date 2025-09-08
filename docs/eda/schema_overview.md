# Schema Overview

## Column Definitions

| Column      | Data_Type      | Example_Values                                                                |
|:------------|:---------------|:------------------------------------------------------------------------------|
| Invoice     | string         | 489434, 489434, 489434                                                        |
| StockCode   | string         | 85048, 79323P, 79323W                                                         |
| Description | string         | 15CM CHRISTMAS GLASS BALL 20 LIGHTS, PINK CHERRY LIGHTS,  WHITE CHERRY LIGHTS |
| Quantity    | int64          | 12, 12, 12                                                                    |
| InvoiceDate | datetime64[ns] | 2009-12-01 07:45:00, 2009-12-01 07:45:00, 2009-12-01 07:45:00                 |
| Price       | float64        | 6.95, 6.75, 6.75                                                              |
| Customer ID | float64        | 13085.0, 13085.0, 13085.0                                                     |
| Country     | string         | United Kingdom, United Kingdom, United Kingdom                                |
| source_year | object         | 2009-2010, 2009-2010, 2009-2010                                               |

**Total Columns:** 9
**Total Rows:** 1,067,371

## Notes
- Invoice: Transaction identifier (prefix 'C' indicates credit/return)
- StockCode: Product SKU identifier
- Customer ID: Unique customer identifier (may contain leading zeros)
