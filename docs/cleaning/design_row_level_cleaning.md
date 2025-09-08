# 🧼 Row-Level Cleaning Pipeline – **Comprehensive Documentation**

*(Online Retail II dataset – `pandas` implementation)*

> **Version**: 2025-06-02  **Script**: `row_cleaning.py`  **Python** ≥ 3.9

---

## 📑 Table of Contents

1. [High-Level Overview](#1-high-level-overview)
2. [Prerequisites & Installation](#2-prerequisites--installation)
3. [Configuration Reference](#3-configuration-reference)
4. [Expected Raw Data Schema](#4-expected-raw-data-schema)
5. [Execution Flow](#5-execution-flow)
6. [Functional Breakdown (Step 0 → 12)](#6-functional-breakdown-step-012)
7. [Row Transformation Logic](#7-row-transformation-logic)
8. [Artefacts & Logs](#8-artefacts--logs)
9. [Troubleshooting & FAQ](#9-troubleshooting--faq)
10. [Extensibility Notes](#10-extensibility-notes)

---

## 1  High-Level Overview

|                      |                                                                                                                                                                                                                              |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Goal**             | Convert raw Online Retail II transactions into an analytically safe table: duplicates removed, placeholder rows dropped, credit-note semantics fixed, monetary outliers capped, derived features added, integrity validated. |
| **Input**            | Excel **`online_retail_II.xlsx`** (sheets *Year 2009-2010*, *Year 2010-2011*) – or cached `df_raw_combined.parquet`.                                                                                                         |
| **Output**           | **`orders_clean.parquet`** (fallback **`orders_clean.csv`**). <br/>Auxiliary: `row_loss_waterfall.csv`, `cap_thresholds.yml`, `cleaning_log.md`.                                                                             |
| **Common use cases** | RFM / cohort analytics, customer-journey modelling, inventory KPI dashboards, ML feature engineering, academic reproducibility.                                                                                              |

---

## 2  Prerequisites & Installation

```bash
# Core libraries
pip install pandas numpy scikit-learn pyyaml pyarrow  # pyarrow ≈ parquet engine
# Optional: generate diagrams
pip install diagrams mermaid-cli
```

Python ≥ 3.9 recommended for type-hint support and performance.

---

## 3  Configuration Reference (`config/cleaning.yaml`)

| Key                | Default | Description                                                                                               |
| ------------------ | ------- | --------------------------------------------------------------------------------------------------------- |
| `DROP_MISSING_ID`  | `true`  | `true` → drop rows with missing `Customer_ID`; `false` → impute `anon_<country>` + add `missing_id` flag. |
| `PRICE_CAP_Q`      | `0.98`  | Upper quantile used to cap **`price`** on non-credit rows.                                                |
| `BASKET_CAP_Q`     | `0.99`  | Upper quantile used to cap **`basket_value`** on non-credit rows.                                         |
| `ISOLATION_CONTAM` | `0.0`   | Proportion of multivariate outliers to remove via Isolation Forest. `0` disables Step 11.                 |

> **Tip:** duplicate the YAML under a new name (e.g., `fast_debug.yaml`) to experiment safely.

---

## 4  Expected Raw Data Schema

| Column        | Example                              | Dtype                | Notes                                      |
| ------------- | ------------------------------------ | -------------------- | ------------------------------------------ |
| `Invoice`     | `536365`                             | `object`             | Mixed string / int ➜ coerced to `str`.     |
| `StockCode`   | `85123A`                             | `object`             | Product identifier.                        |
| `Description` | `WHITE HANGING HEART T-LIGHT HOLDER` | `object`             | Free-text; used for placeholder detection. |
| `Quantity`    | `6`                                  | `int64`              | Negatives = returns.                       |
| `InvoiceDate` | `2010-12-01 08:26:00`                | `datetime64[ns]`     | UTC assumed.                               |
| `Price`       | `2.55`                               | `float64`            | Unit price.                                |
| `Customer_ID` | `17850`                              | `float64` (nullable) | Customer identifier.                       |
| `Country`     | `United Kingdom`                     | `object`             | Free-text.                                 |

Additional columns may exist; unused ones are preserved unless filtered by steps.

---

## 5  Execution Flow

```text
main()
 ├─ load_config()
 ├─ load_and_combine_data()      ──┐   (or read cached parquet)
 │                                 │
 │   Sheet 2009-2010  ──┐          │
 │   Sheet 2010-2011  ──┴─ concat  ▼
 ├─ Step-0  normalise column names
 ├─ Step-1  collapse duplicates
 ├─ Step-2  drop placeholders
 ├─ Step-3  drop extreme prices
 ├─ Step-4  drop zero-price non-credit
 ├─ Step-5  fix credit-note prefix
 ├─ Step-6  drop negative non-credit
 ├─ Step-7  handle missing customer_id
 ├─ Step-8  compute basket value
 ├─ Step-9  apply caps
 ├─ Step-10 country bucket
 ├─ Step-10b cancel flag
 ├─ Step-11 isolation forest (optional)
 ├─ Step-12 assertions
 ├─ save cleaned data (parquet or CSV)
 ├─ write waterfall, caps, cleaning log
 └─ print CLI summary
```

*Run*:

```bash
python row_cleaning.py            # uses default YAML
python row_cleaning.py --config config/fast_debug.yaml   # (if CLI wrapper added)
```

---

## 6  Functional Breakdown (Step 0 → 12)

| Step    | Function                             | Purpose                                                                                                    | Input → Output       | Key Log Metrics                            |        |           |                      |              |
| ------- | ------------------------------------ | ---------------------------------------------------------------------------------------------------------- | -------------------- | ------------------------------------------ | ------ | --------- | -------------------- | ------------ |
| **0**   | `step_0_normalize_columns`           | Snake-case column headers.                                                                                 | `df` → `df`          | new column list                            |        |           |                      |              |
| **1**   | `step_1_collapse_duplicates`         | Drop exact duplicates; group by `(invoice, stockcode)` and **sum quantity**, take **first** of other cols. | → `df`, rows removed | duplicates removed                         |        |           |                      |              |
| **2**   | `step_2_drop_technical_placeholders` | Remove rows with \`                                                                                        | quantity             | == 1000`**or** description containing`TEST | MANUAL | ADJUST\`. | → `df`, rows removed | rows removed |
| **3**   | `step_3_drop_extreme_prices`         | Filter rows with `price > 10 000`.                                                                         | → `df`, rows removed | rows removed                               |        |           |                      |              |
| **4**   | `step_4_drop_zero_price_non_credit`  | Remove zero-price rows unless invoice is credit note (`C…`).                                               | …                    | rows removed                               |        |           |                      |              |
| **5**   | `step_5_fix_credit_note_prefix`      | If *all* quantities in an invoice are negative and lacks `'C'`, prepend `'C'`.                             | `df` → `df`          | invoices fixed                             |        |           |                      |              |
| **6**   | `step_6_drop_negative_non_credit`    | Remove negative quantities on non-credit invoices.                                                         | …                    | rows removed                               |        |           |                      |              |
| **7**   | `step_7_handle_missing_customer_id`  | *Drop* or *impute* missing IDs per YAML; optional `missing_id` flag.                                       | …                    | rows affected                              |        |           |                      |              |
| **8**   | `step_8_compute_basket_value`        | `basket_value = quantity × price` for positive quantities.                                                 | `df` → `df`          | —                                          |        |           |                      |              |
| **9**   | `step_9_apply_caps`                  | Cap `price` and `basket_value` at quantile thresholds; return cap dict.                                    | → `df`, caps         | rows capped                                |        |           |                      |              |
| **10**  | `step_10_create_country_bucket`      | Derive `country_bucket ∈ {UK, EU, NonEU, Unknown}`.                                                        | `df` → `df`          | distribution                               |        |           |                      |              |
| **10b** | `step_10b_create_is_cancelled`       | Flag credit notes (`is_cancelled`).                                                                        | `df` → `df`          | cancelled %                                |        |           |                      |              |
| **11**  | `step_11_isolation_forest`           | Optional multivariate outlier removal on *non-credit* rows.                                                | → `df`, rows removed | outliers removed                           |        |           |                      |              |
| **12**  | `step_12_assertions`                 | Enforce invariants: no duplicate `(invoice, stockcode)`, positive prices, date range.                      | `df` → none          | assertion pass                             |        |           |                      |              |

All step functions are pure (no I/O), easing unit testing and orchestration.

---

## 7  Row Transformation Logic

### Ordered Rule Set

```text
For each row r in df:
  0. rename cols → snake_case
  1. if exact duplicate removed earlier (frame-level)
  2. if |qty| = 1000 or desc matches /TEST|MANUAL|ADJUST/  ⇒  DROP
  3. if price > 10 000  ⇒  DROP
  4. if price == 0 and not credit note  ⇒  DROP
  5. if invoice all neg qty and no 'C'  ⇒  prepend 'C' (row value updated)
  6. if qty < 0 and not credit note  ⇒  DROP
  7. if customer_id missing:
         DROP            if config.DROP_MISSING_ID
         else IMPUTE 'anon_<country>' + missing_id=1
  8. basket_value = qty*price           if qty>0 else 0
  9. if price > price_cap              ⇒  price = cap
     if basket_value > basket_cap      ⇒  basket_value = cap
 10. derive country_bucket
 10b. is_cancelled = invoice.startswith('C')
 11. if multivariate outlier (iso-forest) ⇒  DROP
 12. final dataframe must satisfy assertions
```

### Dual Worked Examples

|                  | Raw (sale) | Cleaned                            |   | Raw (credit note) | Cleaned                    |
| ---------------- | ---------- | ---------------------------------- | - | ----------------- | -------------------------- |
| `Invoice`        | `536365`   | `536365`                           |   | `536540`          | `C536540` *(prefix added)* |
| `Quantity`       | `6`        | `6`                                |   | `-2`              | `-2` *(kept; credit note)* |
| `Price`          | `12 000`   | *row dropped* *(Step 3)*           |   | `-1.95`           | `-1.95`                    |
| `Customer_ID`    | `NaN`      | *dropped* or `anon_united_kingdom` |   | `15311`           | `15311`                    |
| `basket_value`   | —          | —                                  |   | —                 | `-3.90`                    |
| `country_bucket` | —          | `UK`                               |   | —                 | `UK`                       |
| `is_cancelled`   | —          | `0`                                |   | —                 | `1`                        |

---

## 8  Artefacts & Logs

| File                     | Location           | What It Contains                                                                |
| ------------------------ | ------------------ | ------------------------------------------------------------------------------- |
| `orders_clean.parquet`   | `data/processed/`  | Final cleaned dataset.                                                          |
| `row_loss_waterfall.csv` | `docs/cleaning/`   | Row count after each step (easy plotting).                                      |
| `cap_thresholds.yml`     | `docs/cleaning/`   | Computed `price_cap`, `basket_cap`, row counts capped.                          |
| `cleaning_log.md`        | `docs/cleaning/`   | Human-readable summary (suitable for audit).                                    |
| Standard log stream      | console + `logger` | INFO-level messages for each step. Use `export LOGLEVEL=DEBUG` for more detail. |

---

## 9  Troubleshooting & FAQ

| Symptom                                              | Likely Cause                                            | Fix                                                                                                 |
| ---------------------------------------------------- | ------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| **`AssertionError: duplicate (invoice, stockcode)`** | Upstream data changed; Step 1 aggregation keys altered. | Inspect duplicates via `df[df.duplicated(['invoice','stockcode'], keep=False)]`.                    |
| **`Parquet engine not available`**                   | `pyarrow` or `fastparquet` missing.                     | `pip install pyarrow`.                                                                              |
| **MemoryError on load**                              | Large dataset + 32-bit Python or low RAM.               | Load via chunks (`pd.read_csv(..., chunksize=...)`) or run on 64-bit Python with sufficient memory. |
| **Isolation Forest removal too aggressive**          | `ISOLATION_CONTAM` too high.                            | Lower contamination (e.g., `0.005`) or set to `0`.                                                  |

---

## 10  Extensibility Notes

* **Orchestration** – Each `step_n_*` is stateless; wrap them in Airflow/Dagster nodes.
* **Spark adaptation** – Replace pandas ops with PySpark equivalents; group-by semantics remain.
* **Unit tests** – Provide a 5-row fixture per edge case and assert post-conditions for each step.
* **CLI flags** – Consider adding `argparse` to override YAML at runtime (`--price-cap-q 0.97`).
* **Docs automation** – Generate this file from docstrings via MkDocs + `mkdocstrings` during CI.

---

