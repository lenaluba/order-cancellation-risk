"""Extended exploratory data analysis for Online Retail II dataset."""

import hashlib
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import pyarrow

warnings.filterwarnings('ignore')

# Configuration
DATA_PATH = Path("data/raw/online_retail_II.xlsx")
# DF_RAW_PARQUET_PATH = Path("data/raw/online_retail_data_2009_2010.parquet"
# METADATA_PICKLE_PATH = Path("data/raw/online_retail_metadata_2009_2010.pickle"
OUTPUT_DIR = Path("docs/eda")
PLOTS_DIR = OUTPUT_DIR / "plots"

# Create output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def calculate_file_hash(file_path: Path) -> str:
    """Calculate SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_data() -> Tuple[pd.DataFrame, Dict]:
    """Load and combine both sheets from Excel file with metadata."""
    print("Loading data from Excel file...")
    
    # Load both sheets
    sheet_2009_2010 = pd.read_excel(DATA_PATH, sheet_name="Year 2009-2010")
    sheet_2010_2011 = pd.read_excel(DATA_PATH, sheet_name="Year 2010-2011")
    
    # Add source year column
    sheet_2009_2010['source_year'] = "2009-2010"
    sheet_2010_2011['source_year'] = "2010-2011"
    
    # Combine sheets
    df_raw = pd.concat([sheet_2009_2010, sheet_2010_2011], ignore_index=True)
    
    # Calculate metadata
    metadata = {
        'source_file_hash': calculate_file_hash(DATA_PATH),
        'row_count': len(df_raw),
        'sheet_2009_2010_rows': len(sheet_2009_2010),
        'sheet_2010_2011_rows': len(sheet_2010_2011)
    }
    
    print(f"Loaded {metadata['row_count']} total rows")
    print(f"- 2009-2010: {metadata['sheet_2009_2010_rows']} rows")
    print(f"- 2010-2011: {metadata['sheet_2010_2011_rows']} rows")
    print(f"File hash: {metadata['source_file_hash']}")
    
    return df_raw, metadata


def generate_schema_overview(df: pd.DataFrame) -> None:
    """Generate schema overview and save to markdown."""
    print("Generating schema overview...")
    
    schema_data = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        non_null_values = df[col].dropna()
        
        if len(non_null_values) >= 3:
            examples = non_null_values.iloc[:3].tolist()
        else:
            examples = non_null_values.tolist()
        
        examples_str = ", ".join([str(x) for x in examples])
        
        schema_data.append({
            'Column': col,
            'Data_Type': dtype,
            'Example_Values': examples_str
        })
    
    schema_df = pd.DataFrame(schema_data)
    
    # Save as markdown
    with open(OUTPUT_DIR / "schema_overview.md", 'w') as f:
        f.write("# Schema Overview\n\n")
        f.write(schema_df.to_markdown(index=False))
        f.write(f"\n\n**Total Columns:** {len(df.columns)}\n")
        f.write(f"**Total Rows:** {len(df)}\n")


def generate_null_counts(df: pd.DataFrame) -> None:
    """Generate null counts analysis."""
    print("Analyzing null counts...")
    
    null_data = []
    for col in df.columns:
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df)) * 100
        
        null_data.append({
            'Column': col,
            'Null_Count': null_count,
            'Null_Percentage': round(null_pct, 2),
            'Non_Null_Count': len(df) - null_count
        })
    
    null_df = pd.DataFrame(null_data)
    null_df.to_csv(OUTPUT_DIR / "null_counts.csv", index=False)


def generate_unique_counts(df: pd.DataFrame) -> None:
    """Generate unique value counts analysis."""
    print("Analyzing unique value counts...")
    
    unique_data = []
    for col in df.columns:
        unique_count = df[col].nunique()
        unique_pct = (unique_count / len(df)) * 100
        
        unique_data.append({
            'Column': col,
            'Unique_Count': unique_count,
            'Unique_Percentage': round(unique_pct, 2),
            'Total_Count': len(df)
        })
    
    unique_df = pd.DataFrame(unique_data)
    unique_df.to_csv(OUTPUT_DIR / "unique_counts.csv", index=False)


def generate_sanity_checks(df: pd.DataFrame) -> None:
    """Perform sanity checks and save results."""
    print("Performing sanity checks...")
    
    # Calculate metrics
    total_rows = len(df)
    customer_id_missing = df['Customer ID'].isnull().sum()
    customer_id_missing_pct = (customer_id_missing / total_rows) * 100
    
    # Non-credit note rows
    non_credit_rows = ~df['Invoice'].astype(str).str.startswith('C')
    
    qty_negative = (df['Quantity'] <= 0) & non_credit_rows
    qty_negative_count = qty_negative.sum()
    qty_negative_pct = (qty_negative_count / total_rows) * 100
    
    price_negative = (df['Price'] <= 0) & non_credit_rows
    price_negative_count = price_negative.sum()
    price_negative_pct = (price_negative_count / total_rows) * 100
    
    # Credit notes
    credit_notes = df['Invoice'].astype(str).str.startswith('C')
    credit_notes_count = credit_notes.sum()
    credit_notes_pct = (credit_notes_count / total_rows) * 100
    
    # Duplicates
    duplicated_rows = df.duplicated()
    duplicated_count = duplicated_rows.sum()
    duplicated_pct = (duplicated_count / total_rows) * 100
    
    # Date range
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    date_min = df['InvoiceDate'].min()
    date_max = df['InvoiceDate'].max()
    
    # Country stats
    country_counts = df['Country'].value_counts().head(10)
    total_countries = df['Country'].nunique()
    
    # Save results
    with open(OUTPUT_DIR / "sanity_checks.md", 'w') as f:
        f.write("# Sanity Checks\n\n")
        f.write("## Missing Data\n")
        f.write(f"- **Customer ID missing:** {customer_id_missing:,} rows ({customer_id_missing_pct:.2f}%)\n\n")
        
        f.write("## Data Quality Issues\n")
        f.write(f"- **Quantity <= 0 (non-credit notes):** {qty_negative_count:,} rows ({qty_negative_pct:.2f}%)\n")
        f.write(f"- **Price <= 0 (non-credit notes):** {price_negative_count:,} rows ({price_negative_pct:.2f}%)\n\n")
        
        f.write("## Credit Notes\n")
        f.write(f"- **Invoice starting with 'C':** {credit_notes_count:,} rows ({credit_notes_pct:.2f}%)\n\n")
        
        f.write("## Duplicates\n")
        f.write(f"- **Duplicated rows:** {duplicated_count:,} rows ({duplicated_pct:.2f}%)\n\n")
        
        f.write("## Date Range\n")
        f.write(f"- **Invoice date range:** {date_min.strftime('%Y-%m-%d')} to {date_max.strftime('%Y-%m-%d')}\n\n")
        
        f.write("## Geographic Distribution\n")
        f.write(f"- **Total countries:** {total_countries}\n")
        f.write("- **Top 10 countries by row count:**\n")
        for country, count in country_counts.items():
            pct = (count / total_rows) * 100
            f.write(f"  - {country}: {count:,} rows ({pct:.2f}%)\n")


def plot_distribution_analysis(df: pd.DataFrame, column: str, feature_name: str) -> Dict:
    """Create distribution plots and perform normality tests."""
    print(f"Analyzing distribution for {feature_name}...")
    
    # Remove infinite and null values
    clean_data = df[column].replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(clean_data) == 0:
        return {'shapiro_pvalue': None, 'note': 'No valid data'}
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Distribution Analysis: {feature_name}', fontsize=16)
    
    # Histogram (raw)
    axes[0, 0].hist(clean_data, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Histogram (Raw Scale)')
    axes[0, 0].set_xlabel(feature_name)
    axes[0, 0].set_ylabel('Frequency')
    
    # Histogram (log scale) - only for positive values
    positive_data = clean_data[clean_data > 0]
    if len(positive_data) > 0:
        axes[0, 1].hist(np.log10(positive_data), bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Histogram (Log10 Scale)')
        axes[0, 1].set_xlabel(f'Log10({feature_name})')
        axes[0, 1].set_ylabel('Frequency')
    else:
        axes[0, 1].text(0.5, 0.5, 'No positive values\nfor log scale', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Histogram (Log10 Scale) - No Data')
    
    # Boxplot
    axes[1, 0].boxplot(clean_data)
    axes[1, 0].set_title('Boxplot')
    axes[1, 0].set_ylabel(feature_name)
    
    # QQ plot
    if len(clean_data) > 3:  # Need at least 3 points for QQ plot
        stats.probplot(clean_data, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot (Normal Distribution)')
    else:
        axes[1, 1].text(0.5, 0.5, 'Insufficient data\nfor Q-Q plot', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Q-Q Plot - Insufficient Data')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{feature_name.lower().replace(' ', '_')}_distribution.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Separate plots for individual analysis
    # Histogram
    plt.figure(figsize=(8, 6))
    plt.hist(clean_data, bins=50, alpha=0.7, edgecolor='black')
    plt.title(f'{feature_name} - Histogram')
    plt.xlabel(feature_name)
    plt.ylabel('Frequency')
    plt.savefig(PLOTS_DIR / f"{feature_name.lower().replace(' ', '_')}_hist.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Boxplot
    plt.figure(figsize=(8, 6))
    plt.boxplot(clean_data)
    plt.title(f'{feature_name} - Boxplot')
    plt.ylabel(feature_name)
    plt.savefig(PLOTS_DIR / f"{feature_name.lower().replace(' ', '_')}_boxplot.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # QQ plot
    plt.figure(figsize=(8, 6))
    if len(clean_data) > 3:
        stats.probplot(clean_data, dist="norm", plot=plt)
        plt.title(f'{feature_name} - Q-Q Plot')
    else:
        plt.text(0.5, 0.5, 'Insufficient data for Q-Q plot', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title(f'{feature_name} - Q-Q Plot (No Data)')
    plt.savefig(PLOTS_DIR / f"{feature_name.lower().replace(' ', '_')}_qq.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Shapiro-Wilk test (sample if too large)
    test_data = clean_data
    if len(test_data) > 5000:
        test_data = clean_data.sample(5000, random_state=42)
    
    if len(test_data) >= 3:
        try:
            shapiro_stat, shapiro_pvalue = stats.shapiro(test_data)
            return {'shapiro_pvalue': shapiro_pvalue, 'shapiro_stat': shapiro_stat}
        except Exception as e:
            return {'shapiro_pvalue': None, 'note': f'Shapiro test failed: {str(e)}'}
    else:
        return {'shapiro_pvalue': None, 'note': 'Insufficient data for Shapiro test'}


def perform_distribution_analysis(df: pd.DataFrame) -> None:
    """Perform distribution analysis for key variables."""
    print("Performing distribution analysis...")
    
    # Calculate basket value per invoice
    basket_df = df.groupby('Invoice').agg({
        'Quantity': 'sum',
        'Price': 'mean',  # Average unit price per invoice
    }).reset_index()
    basket_df['basket_value'] = basket_df['Quantity'] * basket_df['Price']
    
    # Add basket value back to main dataframe for individual rows
    df_with_basket = df.merge(
        basket_df[['Invoice', 'basket_value']], 
        on='Invoice', 
        how='left'
    )
    
    # Variables to analyze
    variables = [
        ('Quantity', 'Quantity'),
        ('Price', 'Unit Price'),
        ('basket_value', 'Basket Value')
    ]
    
    normality_results = []
    
    for col, name in variables:
        result = plot_distribution_analysis(df_with_basket, col, name)
        normality_results.append({
            'Variable': name,
            'Shapiro_P_Value': result.get('shapiro_pvalue'),
            'Shapiro_Statistic': result.get('shapiro_stat'),
            'Notes': result.get('note', '')
        })
    
    # Save normality test results
    with open(OUTPUT_DIR / "normality_tests.md", 'w') as f:
        f.write("# Normality Tests\n\n")
        f.write("## Shapiro-Wilk Test Results\n\n")
        f.write("| Variable | P-Value | Statistic | Normal? | Notes |\n")
        f.write("|----------|---------|-----------|---------|-------|\n")
        
        for result in normality_results:
            p_val = result['Shapiro_P_Value']
            stat = result['Shapiro_Statistic']
            
            if p_val is not None:
                normal_status = "No" if p_val < 0.05 else "Yes"
                p_val_str = f"{p_val:.6f}"
                stat_str = f"{stat:.6f}" if stat is not None else "N/A"
            else:
                normal_status = "N/A"
                p_val_str = "N/A"
                stat_str = "N/A"
            
            f.write(f"| {result['Variable']} | {p_val_str} | {stat_str} | {normal_status} | {result['Notes']} |\n")
        
        f.write("\n**Note:** P-value < 0.05 indicates non-normal distribution.\n")
        f.write("For large datasets (>5000 rows), tests were performed on a random sample of 5000 observations.\n")


def detect_anomalies(df: pd.DataFrame) -> None:
    """Detect and flag technical value anomalies."""
    print("Detecting anomalies...")
    
    anomaly_results = []
    
    # 1. Quantity == 1,000 or -1,000
    qty_extreme = (df['Quantity'] == 1000) | (df['Quantity'] == -1000)
    anomaly_results.append({
        'condition': 'Quantity == ±1,000',
        'count': qty_extreme.sum(),
        'percentage': (qty_extreme.sum() / len(df)) * 100,
        'examples': df[qty_extreme].head(5) if qty_extreme.any() else pd.DataFrame()
    })
    
    # 2. Price > 10,000
    price_extreme = df['Price'] > 10000
    anomaly_results.append({
        'condition': 'Price > 10,000',
        'count': price_extreme.sum(),
        'percentage': (price_extreme.sum() / len(df)) * 100,
        'examples': df[price_extreme].head(5) if price_extreme.any() else pd.DataFrame()
    })
    
    # 3. Quantity < 0 and Invoice does not start with "C"
    qty_neg_non_credit = (df['Quantity'] < 0) & (~df['Invoice'].astype(str).str.startswith('C'))
    anomaly_results.append({
        'condition': 'Quantity < 0 and Invoice not starting with "C"',
        'count': qty_neg_non_credit.sum(),
        'percentage': (qty_neg_non_credit.sum() / len(df)) * 100,
        'examples': df[qty_neg_non_credit].head(5) if qty_neg_non_credit.any() else pd.DataFrame()
    })
    
    # 4. Price == 0 and Invoice does not start with "C"
    price_zero_non_credit = (df['Price'] == 0) & (~df['Invoice'].astype(str).str.startswith('C'))
    anomaly_results.append({
        'condition': 'Price == 0 and Invoice not starting with "C"',
        'count': price_zero_non_credit.sum(),
        'percentage': (price_zero_non_credit.sum() / len(df)) * 100,
        'examples': df[price_zero_non_credit].head(5) if price_zero_non_credit.any() else pd.DataFrame()
    })
    
    # 5. Descriptions containing TEST, MANUAL, ADJUST (case-insensitive)
    test_keywords = ['TEST', 'MANUAL', 'ADJUST']
    desc_anomaly = (df['Description']
               .astype(str)  # Explicit string conversion
               .fillna('')   # Handle NaN values
               .str.upper()
               .str.contains('|'.join(test_keywords))
               .fillna(False))  # Handle any remaining NaN in boolean mask
    anomaly_results.append({
        'condition': 'Description contains TEST/MANUAL/ADJUST',
        'count': desc_anomaly.sum(),
        'percentage': (desc_anomaly.sum() / len(df)) * 100,
        'examples': df[desc_anomaly].head(5) if desc_anomaly.any() else pd.DataFrame()
    })
    
    # Save results
    with open(OUTPUT_DIR / "anomaly_flags.md", 'w') as f:
        f.write("# Technical Value Anomaly Detection\n\n")
        
        for i, result in enumerate(anomaly_results, 1):
            f.write(f"## {i}. {result['condition']}\n\n")
            f.write(f"**Count:** {result['count']:,} rows\n")
            f.write(f"**Percentage:** {result['percentage']:.4f}%\n\n")
            
            if not result['examples'].empty:
                f.write("**Example rows:**\n")
                f.write(result['examples'].to_markdown(index=False))
                f.write("\n\n")
            else:
                f.write("**No examples found.**\n\n")
        
        # Summary
        total_flagged = sum(result['count'] for result in anomaly_results)
        f.write(f"## Summary\n\n")
        f.write(f"**Total flagged rows:** {total_flagged:,}\n")
        f.write(f"**Total dataset rows:** {len(df):,}\n")
        f.write(f"**Overall anomaly rate:** {(total_flagged / len(df)) * 100:.4f}%\n")


def analyze_basket_values(df: pd.DataFrame) -> None:
    """Analyze basket values and suggest outlier thresholds."""
    print("Analyzing basket values...")
    
    # Calculate basket value per invoice (only positive quantities)
    positive_qty = df[df['Quantity'] > 0].copy()
    positive_qty['line_value'] = positive_qty['Quantity'] * positive_qty['Price']
    
    invoice_df = positive_qty.groupby('Invoice').agg({
        'line_value': 'sum',
        'Quantity': 'sum',
        'Customer ID': 'first',
        'Country': 'first',
        'InvoiceDate': 'first'
    }).reset_index()
    
    invoice_df.rename(columns={'line_value': 'basket_value'}, inplace=True)
    
    # Remove infinite and null values
    clean_baskets = invoice_df['basket_value'].replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(clean_baskets) == 0:
        print("No valid basket values found!")
        return
    
    # Calculate statistics
    percentiles = [99, 99.5, 99.9]
    percentile_values = [np.percentile(clean_baskets, p) for p in percentiles]
    top_5_values = clean_baskets.nlargest(5).tolist()
    
    # Create histogram and boxplot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    axes[0].hist(clean_baskets, bins=50, alpha=0.7, edgecolor='black')
    axes[0].set_title('Basket Value Distribution')
    axes[0].set_xlabel('Basket Value')
    axes[0].set_ylabel('Frequency')
    
    # Boxplot
    axes[1].boxplot(clean_baskets)
    axes[1].set_title('Basket Value Boxplot')
    axes[1].set_ylabel('Basket Value')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "basket_value_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Suggest capping thresholds
    q99 = percentile_values[0]
    q995 = percentile_values[1]
    q999 = percentile_values[2]
    
    # Save results
    with open(OUTPUT_DIR / "outlier_summary.md", 'w') as f:
        f.write("# Basket Value Analysis and Outlier Summary\n\n")
        
        f.write("## Basic Statistics\n")
        f.write(f"- **Total invoices analyzed:** {len(clean_baskets):,}\n")
        f.write(f"- **Mean basket value:** £{clean_baskets.mean():.2f}\n")
        f.write(f"- **Median basket value:** £{clean_baskets.median():.2f}\n")
        f.write(f"- **Standard deviation:** £{clean_baskets.std():.2f}\n\n")
        
        f.write("## Percentile Analysis\n")
        for p, val in zip(percentiles, percentile_values):
            f.write(f"- **{p}th percentile:** £{val:.2f}\n")
        f.write("\n")
        
        f.write("## Top 5 Highest Basket Values\n")
        for i, val in enumerate(top_5_values, 1):
            f.write(f"{i}. £{val:.2f}\n")
        f.write("\n")
        
        f.write("## Suggested Capping Thresholds\n")
        f.write(f"- **Conservative (99th percentile):** £{q99:.2f}\n")
        f.write(f"  - This would cap {(clean_baskets > q99).sum():,} invoices ({((clean_baskets > q99).sum() / len(clean_baskets) * 100):.2f}%)\n")
        f.write(f"- **Moderate (99.5th percentile):** £{q995:.2f}\n")
        f.write(f"  - This would cap {(clean_baskets > q995).sum():,} invoices ({((clean_baskets > q995).sum() / len(clean_baskets) * 100):.2f}%)\n")
        f.write(f"- **Liberal (99.9th percentile):** £{q999:.2f}\n")
        f.write(f"  - This would cap {(clean_baskets > q999).sum():,} invoices ({((clean_baskets > q999).sum() / len(clean_baskets) * 100):.2f}%)\n\n")
        
        f.write("## Recommendation\n")
        f.write("Based on the distribution analysis, we recommend using the **99.5th percentile** ")
        f.write(f"(£{q995:.2f}) as the capping threshold. This strikes a balance between ")
        f.write("preserving legitimate high-value transactions while removing extreme outliers ")
        f.write("that could negatively impact model performance.\n")


def main():
    """Run the complete EDA pipeline."""
    print("=" * 60)
    print("EXTENDED EDA PIPELINE - ONLINE RETAIL II DATASET")
    print("=" * 60)
    
    # Load data
    df_raw, metadata = load_data()

    # Write df_raw to parquet files with snappy compression
    # sheet_2009_2010.to_parquet(
    #     DF_RAW_PARQUET_PATH,
    #     compression='snappy',
    #     index=False
    # )

    # Write metadata to a pickle file
    # with open(METADATA_PICKLE_PATH, 'wb') as f:
    # pickle.dump(metadata, f)
    
    # Basic data summary
    generate_schema_overview(df_raw)
    generate_null_counts(df_raw)
    generate_unique_counts(df_raw)
    
    # Sanity checks
    generate_sanity_checks(df_raw)
    
    # Distribution analysis
    perform_distribution_analysis(df_raw)
    
    # Anomaly detection
    detect_anomalies(df_raw)
    
    # Basket value analysis
    analyze_basket_values(df_raw)
    
    print("\n" + "=" * 60)
    print("EDA PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print(f"All plots saved to: {PLOTS_DIR}")
    print(f"Total files processed: {metadata['row_count']:,} rows")
    print(f"Source file hash: {metadata['source_file_hash'][:16]}...")


if __name__ == "__main__":
    main()