"""
Row-level cleaning script for Online Retail II dataset.
Performs 13-step cleaning sequence and logs all transformations.
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, Tuple, List
import pandas as pd
import numpy as np
import yaml
from sklearn.ensemble import IsolationForest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# EU country list for country bucketing
EU_COUNTRIES = [
    'Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic',
    'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary',
    'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg', 'Malta',
    'Netherlands', 'Poland', 'Portugal', 'Romania', 'Slovakia', 'Slovenia',
    'Spain', 'Sweden', 'EIRE', 'European Community', 'Channel Islands'
]


def load_config(config_path: Path) -> Dict:
    """Load cleaning configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config: {config}")
    return config


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lower snake_case."""
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    logger.info(f"Normalized column names: {list(df.columns)}")
    return df


def load_and_combine_data(excel_path: Path) -> pd.DataFrame:
    """Load Excel file and combine sheets into single dataframe."""
    logger.info(f"Loading data from {excel_path}")
    
    # Read both sheets
    df_2010 = pd.read_excel(excel_path, sheet_name='Year 2010-2011')
    df_2009 = pd.read_excel(excel_path, sheet_name='Year 2009-2010')
    
    logger.info(f"Sheet 'Year 2010-2011': {len(df_2010)} rows")
    logger.info(f"Sheet 'Year 2009-2010': {len(df_2009)} rows")
    
    # Combine sheets
    df_combined = pd.concat([df_2009, df_2010], ignore_index=True)
    logger.info(f"Combined data: {len(df_combined)} rows")
    
    # Ensure columns with mixed types are properly converted to strings
    # This prevents pyarrow serialization errors
    string_columns = ['Invoice', 'StockCode', 'Description', 'Country']
    for col in string_columns:
        if col in df_combined.columns:
            df_combined[col] = df_combined[col].astype(str)
    
    return df_combined


def step_0_normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Step 0: Normalize column names."""
    df = normalize_column_names(df)
    return df


def step_1_collapse_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Step 1: Collapse duplicate rows and aggregate by invoice-stock pairs."""
    initial_rows = len(df)
    
    # Remove exact duplicates
    df = df.drop_duplicates()
    exact_dups_removed = initial_rows - len(df)
    logger.info(f"Removed {exact_dups_removed} exact duplicate rows")
    
    # Aggregate rows with same (invoice, stockcode) by summing quantity
    agg_cols = {col: 'first' for col in df.columns if col not in ['quantity']}
    agg_cols['quantity'] = 'sum'
    
    df = df.groupby(['invoice', 'stockcode'], as_index=False).agg(agg_cols)
    aggregated_rows = len(df)
    
    total_removed = initial_rows - aggregated_rows
    logger.info(f"Step 1 complete: removed {total_removed} total duplicate/aggregated rows")
    
    return df, total_removed


def step_2_drop_technical_placeholders(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Step 2: Drop technical placeholder rows."""
    initial_rows = len(df)
    
    # Drop rows with abs(quantity) == 1000
    mask_quantity = df['quantity'].abs() == 1000
    
    # Drop rows with description containing TEST|MANUAL|ADJUST (case-insensitive)
    mask_description = df['description'].str.contains(
        r'TEST|MANUAL|ADJUST', case=False, na=False, regex=True
    )
    
    mask_combined = mask_quantity | mask_description
    df = df[~mask_combined]
    
    removed = initial_rows - len(df)
    logger.info(f"Step 2: removed {removed} technical placeholder rows")
    
    return df, removed


def step_3_drop_extreme_prices(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Step 3: Drop rows where price > 10,000."""
    initial_rows = len(df)
    df = df[df['price'] <= 10000]
    removed = initial_rows - len(df)
    logger.info(f"Step 3: removed {removed} rows with price > 10,000")
    return df, removed


def step_4_drop_zero_price_non_credit(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Step 4: Drop rows where price == 0 and not a credit note."""
    initial_rows = len(df)
    mask = (df['price'] == 0) & (~df['invoice'].astype(str).str.startswith('C'))
    df = df[~mask]
    removed = initial_rows - len(df)
    logger.info(f"Step 4: removed {removed} rows with price=0 and non-credit invoice")
    return df, removed


def step_5_fix_credit_note_prefix(df: pd.DataFrame) -> pd.DataFrame:
    """Step 5: Fix invoices where all lines are negative but lack 'C' prefix."""
    # Group by invoice and check if all quantities are negative
    invoice_all_negative = df.groupby('invoice')['quantity'].apply(lambda x: (x < 0).all())
    
    # Find invoices that should have 'C' prefix but don't
    invoices_to_fix = invoice_all_negative[invoice_all_negative].index
    invoices_to_fix = [inv for inv in invoices_to_fix 
                       if not str(inv).startswith('C')]
    
    # Add 'C' prefix to these invoices
    if invoices_to_fix:
        df.loc[df['invoice'].isin(invoices_to_fix), 'invoice'] = (
            'C' + df.loc[df['invoice'].isin(invoices_to_fix), 'invoice'].astype(str)
        )
        logger.info(f"Step 5: added 'C' prefix to {len(invoices_to_fix)} invoices")
    else:
        logger.info("Step 5: no invoices needed 'C' prefix correction")
    
    return df


def step_6_drop_negative_non_credit(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Step 6: Delete rows where quantity < 0 and invoice is not credit note."""
    initial_rows = len(df)
    mask = (df['quantity'] < 0) & (~df['invoice'].astype(str).str.startswith('C'))
    df = df[~mask]
    removed = initial_rows - len(df)
    logger.info(f"Step 6: removed {removed} rows with negative quantity in non-credit invoices")
    return df, removed


def step_7_handle_missing_customer_id(df: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, int]:
    """Step 7: Handle missing customer IDs based on config."""
    missing_count = df['customer_id'].isna().sum()
    
    if config['DROP_MISSING_ID']:
        initial_rows = len(df)
        df = df.dropna(subset=['customer_id'])
        removed = initial_rows - len(df)
        logger.info(f"Step 7: dropped {removed} rows with missing customer_id")
    else:
        # Create surrogate IDs
        mask = df['customer_id'].isna()
        # Convert customer_id to string type first to avoid dtype warning
        df['customer_id'] = df['customer_id'].astype('object')
        df.loc[mask, 'customer_id'] = (
            'anon_' + df.loc[mask, 'country'].str.lower().fillna('unknown')
        )
        # Add missing_id indicator column
        df['missing_id'] = mask.astype(int)
        logger.info(f"Step 7: imputed {missing_count} missing customer_ids with anonymous IDs")
        removed = 0
    
    return df, removed


def step_8_compute_basket_value(df: pd.DataFrame) -> pd.DataFrame:
    """Step 8: Compute basket value for positive quantity lines."""
    # Only compute for positive quantities
    mask = df['quantity'] > 0
    df.loc[mask, 'basket_value'] = df.loc[mask, 'quantity'] * df.loc[mask, 'price']
    df.loc[~mask, 'basket_value'] = 0
    logger.info("Step 8: computed basket_value for positive quantity lines")
    return df


def step_9_apply_caps(df: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, Dict]:
    """Step 9: Apply price and basket value caps based on quantiles."""
    # Calculate caps for non-credit rows only
    non_credit_mask = ~df['invoice'].astype(str).str.startswith('C')
    
    price_cap = df.loc[non_credit_mask, 'price'].quantile(config['PRICE_CAP_Q'])
    basket_cap = df.loc[non_credit_mask & (df['basket_value'] > 0), 'basket_value'].quantile(
        config['BASKET_CAP_Q']
    )
    
    # Apply caps
    price_capped = (df['price'] > price_cap).sum()
    df.loc[df['price'] > price_cap, 'price'] = price_cap
    
    basket_capped = (df['basket_value'] > basket_cap).sum()
    df.loc[df['basket_value'] > basket_cap, 'basket_value'] = basket_cap
    
    caps = {
        'price_cap': float(price_cap),
        'basket_cap': float(basket_cap),
        'price_capped_count': int(price_capped),
        'basket_capped_count': int(basket_capped)
    }
    
    logger.info(f"Step 9: applied caps - price_cap={price_cap:.2f} ({price_capped} rows), "
                f"basket_cap={basket_cap:.2f} ({basket_capped} rows)")
    
    return df, caps


def step_10_create_country_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """Step 10: Create country bucket column."""
    def categorize_country(country):
        if pd.isna(country):
            return 'Unknown'
        elif country == 'United Kingdom':
            return 'UK'
        elif country in EU_COUNTRIES:
            return 'EU'
        else:
            return 'NonEU'
    
    df['country_bucket'] = df['country'].apply(categorize_country)
    
    # Log distribution
    distribution = df['country_bucket'].value_counts()
    logger.info(f"Step 10: created country_bucket - {distribution.to_dict()}")
    
    return df


def step_10b_create_is_cancelled(df: pd.DataFrame) -> pd.DataFrame:
    """Step 10b: Create is_cancelled indicator column."""
    # Create binary indicator for cancelled orders (credit notes)
    df['is_cancelled'] = df['invoice'].astype(str).str.startswith('C').astype(int)
    
    # Log distribution
    cancelled_count = df['is_cancelled'].sum()
    total_count = len(df)
    logger.info(f"Step 10b: created is_cancelled indicator - {cancelled_count:,} cancelled orders "
                f"({cancelled_count/total_count*100:.1f}% of total)")
    
    return df


def step_11_isolation_forest(df: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, int]:
    """Step 11: Optional multivariate outlier detection using Isolation Forest."""
    if config.get('ISOLATION_CONTAM', 0) == 0:
        logger.info("Step 11: skipped (ISOLATION_CONTAM = 0)")
        return df, 0
    
    # Select features for outlier detection (non-credit rows only)
    non_credit_mask = ~df['invoice'].astype(str).str.startswith('C')
    features = ['quantity', 'price', 'basket_value']
    
    # Prepare data
    X = df.loc[non_credit_mask, features].fillna(0)
    
    # Fit Isolation Forest
    iso_forest = IsolationForest(
        contamination=config['ISOLATION_CONTAM'],
        random_state=42,
        n_estimators=100
    )
    
    # Predict outliers (-1 for outliers, 1 for inliers)
    outlier_predictions = iso_forest.fit_predict(X)
    
    # Create outlier mask for full dataframe
    outlier_mask = pd.Series(False, index=df.index)
    outlier_mask[non_credit_mask] = outlier_predictions == -1
    
    # Remove outliers
    initial_rows = len(df)
    df = df[~outlier_mask]
    removed = initial_rows - len(df)
    
    logger.info(f"Step 11: removed {removed} multivariate outliers using Isolation Forest")
    
    return df, removed


def step_12_assertions(df: pd.DataFrame) -> None:
    """Step 12: Run assertions to validate cleaned data."""
    # Check for duplicates
    duplicates = df.duplicated(subset=['invoice', 'stockcode']).sum()
    assert duplicates == 0, f"Found {duplicates} duplicate (invoice, stockcode) pairs"
    
    # Check no price <= 0 on non-credit rows
    non_credit_mask = ~df['invoice'].astype(str).str.startswith('C')
    invalid_prices = (df.loc[non_credit_mask, 'price'] <= 0).sum()
    assert invalid_prices == 0, f"Found {invalid_prices} non-credit rows with price <= 0"
    
    # Check date range - get actual min/max for logging
    actual_min = df['invoicedate'].min()
    actual_max = df['invoicedate'].max()
    min_date = pd.to_datetime('2009-12-01')
    max_date = pd.to_datetime('2011-12-09 23:59:59')  # Include full day
    
    # Log actual date range
    logger.info(f"Actual date range: {actual_min} to {actual_max}")
    
    date_range_valid = (
        (df['invoicedate'] >= min_date) & 
        (df['invoicedate'] <= max_date)
    ).all()
    assert date_range_valid, f"Found dates outside expected range (2009-12-01 to 2011-12-09). Actual range: {actual_min} to {actual_max}"
    
    logger.info("Step 12: all assertions passed")


def create_waterfall_log(steps: List[Dict]) -> pd.DataFrame:
    """Create waterfall log of row losses."""
    waterfall = pd.DataFrame(steps)
    return waterfall


def create_cleaning_log(df_initial: pd.DataFrame, df_final: pd.DataFrame, 
                       steps_info: Dict, caps: Dict, config: Dict) -> str:
    """Create detailed cleaning log."""
    log_content = f"""# Row-Level Cleaning Log

## Summary
- Initial rows: {len(df_initial):,}
- Final rows: {len(df_final):,}
- Retention rate: {len(df_final) / len(df_initial) * 100:.1f}%

## Cleaning Steps

### Step 1: Duplicate Removal
- Exact duplicates removed: {steps_info.get('exact_dups', 0):,}
- Rows aggregated by (invoice, stockcode): {steps_info.get('step_1', 0):,}

### Step 2: Technical Placeholders
- Rows with |quantity| = 1000: removed
- Rows with TEST/MANUAL/ADJUST in description: removed
- Total removed: {steps_info.get('step_2', 0):,}

### Step 3: Extreme Prices
- Rows with price > 10,000: {steps_info.get('step_3', 0):,}

### Step 4: Zero Price Non-Credit
- Rows removed: {steps_info.get('step_4', 0):,}

### Step 5: Credit Note Prefix Correction
- Invoices corrected: {steps_info.get('invoices_fixed', 0):,}

### Step 6: Negative Quantity Non-Credit
- Rows removed: {steps_info.get('step_6', 0):,}

### Step 7: Missing Customer ID
- Policy: {'DROP' if config['DROP_MISSING_ID'] else 'IMPUTE'}
- Rows affected: {steps_info.get('step_7', 0):,}

### Step 8: Basket Value Computation
- Computed for all positive quantity rows

### Step 9: Value Capping
- Price cap (Q{config['PRICE_CAP_Q']}): £{caps['price_cap']:.2f}
- Basket cap (Q{config['BASKET_CAP_Q']}): £{caps['basket_cap']:.2f}
- Prices capped: {caps['price_capped_count']:,}
- Baskets capped: {caps['basket_capped_count']:,}

### Step 10: Country Bucketing
- Categories: UK, EU, NonEU

### Step 11: Multivariate Outlier Detection
- Method: Isolation Forest
- Contamination: {config['ISOLATION_CONTAM']}
- Outliers removed: {steps_info.get('step_11', 0):,}

### Step 12: Final Assertions
- No duplicate (invoice, stockcode) pairs: ✓
- No price <= 0 on non-credit rows: ✓
- Date range within 2009-12-01 to 2011-12-09: ✓

## Final Dataset
- Rows: {len(df_final):,}
- Columns: {len(df_final.columns)}
- Memory usage: {df_final.memory_usage(deep=True).sum() / 1024**2:.1f} MB
- Key derived columns: is_cancelled (1 for credit notes), country_bucket, basket_value
"""
    return log_content


def main():
    """Main cleaning pipeline."""
    # Setup paths
    project_root = Path(__file__).parent.parent
    excel_path = project_root / "data" / "raw" / "online_retail_II.xlsx"
    interim_path = project_root / "data" / "interim" / "df_raw_combined.parquet"
    output_path = project_root / "data" / "processed" / "orders_clean.parquet"
    config_path = project_root / "config" / "cleaning.yaml"
    
    # Load config
    config = load_config(config_path)
    
    # Track steps for waterfall
    waterfall_data = []
    steps_info = {}
    
    # Load data
    if interim_path.exists():
        logger.info(f"Loading interim parquet from {interim_path}")
        df = pd.read_parquet(interim_path)
    else:
        logger.info("Interim parquet not found, loading from Excel")
        df = load_and_combine_data(excel_path)
        # Try to save interim file for future use
        try:
            interim_path.parent.mkdir(exist_ok=True)
            # Use strings_to_categorical=False to handle mixed types better
            df.to_parquet(interim_path, engine='pyarrow', 
                         coerce_timestamps='ms', allow_truncated_timestamps=True)
            logger.info(f"Saved interim data to {interim_path}")
        except ImportError:
            logger.warning("Parquet engine not available, skipping interim save")
    
    df_initial = df.copy()
    initial_rows = len(df)
    waterfall_data.append({
        'step': 'initial',
        'rows_remaining': initial_rows,
        'rows_removed': 0
    })
    
    # Step 0: Normalize columns
    df = step_0_normalize_columns(df)
    
    # Step 1: Collapse duplicates
    df, removed = step_1_collapse_duplicates(df)
    steps_info['step_1'] = removed
    waterfall_data.append({
        'step': 'collapse_duplicates',
        'rows_remaining': len(df),
        'rows_removed': removed
    })
    
    # Step 2: Drop technical placeholders
    df, removed = step_2_drop_technical_placeholders(df)
    steps_info['step_2'] = removed
    waterfall_data.append({
        'step': 'drop_technical',
        'rows_remaining': len(df),
        'rows_removed': removed
    })
    
    # Step 3: Drop extreme prices
    df, removed = step_3_drop_extreme_prices(df)
    steps_info['step_3'] = removed
    waterfall_data.append({
        'step': 'drop_extreme_prices',
        'rows_remaining': len(df),
        'rows_removed': removed
    })
    
    # Step 4: Drop zero price non-credit
    df, removed = step_4_drop_zero_price_non_credit(df)
    steps_info['step_4'] = removed
    waterfall_data.append({
        'step': 'drop_zero_price',
        'rows_remaining': len(df),
        'rows_removed': removed
    })
    
    # Step 5: Fix credit note prefix
    df = step_5_fix_credit_note_prefix(df)
    
    # Step 6: Drop negative non-credit
    df, removed = step_6_drop_negative_non_credit(df)
    steps_info['step_6'] = removed
    waterfall_data.append({
        'step': 'drop_negative_non_credit',
        'rows_remaining': len(df),
        'rows_removed': removed
    })
    
    # Step 7: Handle missing customer ID
    df, removed = step_7_handle_missing_customer_id(df, config)
    steps_info['step_7'] = removed
    waterfall_data.append({
        'step': 'handle_missing_id',
        'rows_remaining': len(df),
        'rows_removed': removed
    })
    
    # Step 8: Compute basket value
    df = step_8_compute_basket_value(df)
    
    # Step 9: Apply caps
    df, caps = step_9_apply_caps(df, config)
    
    # Step 10: Create country bucket
    df = step_10_create_country_bucket(df)
    
    # Step 10b: Create is_cancelled indicator
    df = step_10b_create_is_cancelled(df)
    
    # Step 11: Isolation Forest
    df, removed = step_11_isolation_forest(df, config)
    steps_info['step_11'] = removed
    waterfall_data.append({
        'step': 'isolation_forest',
        'rows_remaining': len(df),
        'rows_removed': removed
    })
    
    # Step 12: Assertions
    step_12_assertions(df)
    
    # Save outputs
    output_path.parent.mkdir(exist_ok=True)
    try:
        df.to_parquet(output_path, index=False, engine='pyarrow',
                     coerce_timestamps='ms', allow_truncated_timestamps=True)
        logger.info(f"Saved cleaned data to {output_path}")
    except ImportError:
        # Fallback to CSV if parquet not available
        csv_path = output_path.with_suffix('.csv')
        df.to_csv(csv_path, index=False)
        logger.warning(f"Parquet engine not available, saved as CSV to {csv_path}")
    
    # Create and save waterfall log
    waterfall_df = create_waterfall_log(waterfall_data)
    waterfall_path = project_root / "docs" / "cleaning" / "row_loss_waterfall.csv"
    waterfall_df.to_csv(waterfall_path, index=False)
    logger.info(f"Saved waterfall log to {waterfall_path}")
    
    # Save cap thresholds
    caps_path = project_root / "docs" / "cleaning" / "cap_thresholds.yml"
    with open(caps_path, 'w') as f:
        yaml.dump(caps, f)
    logger.info(f"Saved cap thresholds to {caps_path}")
    
    # Create and save cleaning log
    cleaning_log = create_cleaning_log(df_initial, df, steps_info, caps, config)
    log_path = project_root / "docs" / "cleaning" / "cleaning_log.md"
    with open(log_path, 'w') as f:
        f.write(cleaning_log)
    logger.info(f"Saved cleaning log to {log_path}")
    
    # Print summary
    print(f"\n{'='*50}")
    print("CLEANING SUMMARY")
    print(f"{'='*50}")
    print(f"Cleaned rows: {len(df):,} (retained {len(df)/initial_rows*100:.1f}% of raw)")
    print(f"Duplicates removed: {steps_info.get('step_1', 0):,}")
    print(f"Technical placeholder rows removed: {steps_info.get('step_2', 0):,}")
    if config['DROP_MISSING_ID']:
        print(f"Missing IDs dropped: {steps_info.get('step_7', 0):,}")
    else:
        print(f"Missing IDs imputed: {steps_info.get('step_7', 0):,}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()