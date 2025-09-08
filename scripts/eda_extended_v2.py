"""
Online Retail II Extended Exploratory Data Analysis Pipeline
===========================================================

Business Context:
    - Purpose: Deep-dive analysis of UK online retail transaction patterns for strategic insights
    - Stakeholders: Product Analytics, Revenue Operations, Customer Success teams
    - Key Decisions Supported:
        * Inventory optimization based on purchase patterns
        * Customer segmentation for targeted marketing
        * Pricing strategy adjustments based on basket analysis
        * Data quality thresholds for ML model training

Technical Approach:
    - Comprehensive data profiling with business-relevant anomaly detection
    - Statistical distribution analysis for feature engineering decisions
    - Basket value analysis for outlier treatment recommendations
    - Automated report generation for stakeholder communication

Usage: python extended_eda_pipeline.py
Output: 
    - Markdown reports in docs/eda/
    - Statistical plots in docs/eda/plots/
    - CSV summaries for downstream analysis

Dependencies: pandas>=1.3.0, numpy>=1.19.0, matplotlib>=3.3.0, seaborn>=0.11.0, scipy>=1.7.0
"""

import hashlib
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Suppress non-critical warnings to keep logs clean for business users
warnings.filterwarnings('ignore')

# ========== CONFIGURATION ==========
# Business-driven thresholds and paths - modify for different analysis scenarios

# Data source configuration
DATA_PATH = Path("data/raw/online_retail_II.xlsx")
SHEET_NAMES = ["Year 2009-2010", "Year 2010-2011"]  # Fiscal year boundaries

# Output configuration for report distribution
OUTPUT_DIR = Path("docs/eda")
PLOTS_DIR = OUTPUT_DIR / "plots"

# Business rule thresholds
MIN_ROWS_FOR_VALID_ANALYSIS = 1000  # Below this, data load likely failed
EXTREME_QUANTITY_THRESHOLD = 1000   # Flag potential data entry errors
EXTREME_PRICE_THRESHOLD = 10000     # Flag luxury items or data errors
SHAPIRO_TEST_MAX_SAMPLES = 5000     # Statistical test performance limit

# Anomaly detection keywords - indicates manual adjustments or test data
TECHNICAL_ANOMALY_KEYWORDS = ['TEST', 'MANUAL', 'ADJUST']

# Percentiles for outlier analysis - aligned with business risk tolerance
OUTLIER_PERCENTILES = [99, 99.5, 99.9]

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def calculate_file_hash(file_path: Path) -> str:
    """
    Calculate SHA-256 hash for data lineage and compliance tracking.
    
    Business Purpose:
        - Ensures data integrity for financial reporting
        - Provides audit trail for regulatory compliance
        - Enables detection of unauthorized data modifications
        
    Args:
        file_path: Path to source data file
        
    Returns:
        Hexadecimal SHA-256 hash string for audit logs
        
    Raises:
        FileNotFoundError: Source file missing or inaccessible
    """
    try:
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            # Process in 8KB chunks for memory efficiency with large files
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except FileNotFoundError:
        logger.error(f"Source file not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Failed to calculate file hash: {e}")
        raise


def load_and_combine_fiscal_years() -> Tuple[pd.DataFrame, Dict]:
    """
    Load and combine multi-year retail transaction data with metadata tracking.
    
    Business Purpose:
        - Consolidates fiscal year data for year-over-year analysis
        - Preserves data lineage for financial auditing
        - Enables cohort analysis across time periods
        
    Design Decisions:
        - Concatenates sheets to preserve all transactions (no deduplication)
        - Adds source_year column for temporal cohort analysis
        - Calculates metadata for data quality monitoring
        
    Returns:
        Tuple of (combined_dataframe, metadata_dict)
        
    Raises:
        ValueError: Data integrity issues preventing analysis
        FileNotFoundError: Source Excel file not accessible
    """
    logger.info("Initiating fiscal year data consolidation...")
    
    try:
        # Calculate source file hash for compliance audit trail
        source_hash = calculate_file_hash(DATA_PATH)
        logger.info(f"Source file integrity verified: {source_hash[:16]}...")
        
        # Load fiscal year sheets with progress tracking
        fiscal_year_frames = []
        sheet_metadata = {}
        """
        for sheet_name in SHEET_NAMES:
            logger.info(f"Loading {sheet_name}...")
            
            # Read with default types to preserve data fidelity
            # Note: Using openpyxl engine for .xlsx compatibility
            df_sheet = pd.read_excel(DATA_PATH, sheet_name=sheet_name)
            
            # Validate minimum data requirements
            if len(df_sheet) < MIN_ROWS_FOR_VALID_ANALYSIS:
                logger.warning(f"{sheet_name}: Suspiciously low row count ({len(df_sheet):,})")
            
            # Add source tracking for year-over-year analysis
            df_sheet['source_year'] = sheet_name.replace("Year ", "")
            
            fiscal_year_frames.append(df_sheet)
            sheet_metadata[f'{sheet_name}_rows'] = len(df_sheet)
            
            logger.info(f"Loaded {sheet_name}: {len(df_sheet):,} transactions")
        
        # Combine fiscal years preserving all transactions
        df_combined = pd.concat(fiscal_year_frames, ignore_index=True)
        """
        df_combined = pd.read_parquet ('.//data/interim//df_raw_combined.parquet')
        # Compile metadata for data quality reporting
        metadata = {
            'source_file_hash': source_hash,
            'total_row_count': len(df_combined),
            **sheet_metadata  # Unpack individual sheet counts
        }
        
        # Log summary statistics for monitoring
        logger.info(f"Data consolidation complete: {metadata['total_row_count']:,} total transactions")
        for sheet_name in SHEET_NAMES:
            key = f'{sheet_name}_rows'
            if key in metadata:
                logger.info(f"  - {sheet_name}: {metadata[key]:,} rows")
        
        return df_combined, metadata
        
    except FileNotFoundError:
        logger.error(f"Cannot access source data: {DATA_PATH}")
        raise
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        raise


def generate_schema_documentation(df: pd.DataFrame) -> None:
    """
    Generate human-readable schema documentation for stakeholder review.
    
    Business Purpose:
        - Provides data dictionary for business analysts
        - Documents available fields for feature engineering
        - Shows example values for data validation
        
    Design Decisions:
        - Includes 3 example values per column for context
        - Saves as markdown for easy sharing via documentation portals
        - Preserves data types for technical reference
        
    Args:
        df: Source dataframe to document
    """
    logger.info("Generating schema documentation...")
    
    schema_records = []
    
    for column in df.columns:
        # Extract column metadata
        dtype = str(df[column].dtype)
        non_null_values = df[column].dropna()
        
        # Get representative examples (up to 3)
        if len(non_null_values) >= 3:
            examples = non_null_values.iloc[:3].tolist()
        else:
            examples = non_null_values.tolist()
        
        # Format examples for readability
        examples_formatted = ", ".join([str(x) for x in examples])
        
        schema_records.append({
            'Column': column,
            'Data_Type': dtype,
            'Example_Values': examples_formatted
        })
    
    schema_df = pd.DataFrame(schema_records)
    
    # Generate markdown report
    output_path = OUTPUT_DIR / "schema_overview.md"
    with open(output_path, 'w') as f:
        f.write("# Schema Overview\n\n")
        f.write("## Column Definitions\n\n")
        f.write(schema_df.to_markdown(index=False))
        f.write(f"\n\n**Total Columns:** {len(df.columns)}\n")
        f.write(f"**Total Rows:** {len(df):,}\n")
        f.write("\n## Notes\n")
        f.write("- Invoice: Transaction identifier (prefix 'C' indicates credit/return)\n")
        f.write("- StockCode: Product SKU identifier\n")
        f.write("- Customer ID: Unique customer identifier (may contain leading zeros)\n")
    
    logger.info(f"Schema documentation saved to: {output_path}")


def analyze_data_completeness(df: pd.DataFrame) -> None:
    """
    Analyze missing data patterns for feature engineering decisions.
    
    Business Purpose:
        - Identifies fields requiring imputation strategies
        - Flags potential data collection issues
        - Guides feature selection for ML models
        
    Design Decisions:
        - Calculates both absolute and percentage metrics
        - Exports to CSV for downstream pipeline integration
        - Focuses on business-critical fields
        
    Args:
        df: Dataframe to analyze for completeness
    """
    logger.info("Analyzing data completeness...")
    
    null_analysis = []
    total_rows = len(df)
    
    for column in df.columns:
        null_count = df[column].isnull().sum()
        null_percentage = (null_count / total_rows) * 100
        
        null_analysis.append({
            'Column': column,
            'Null_Count': null_count,
            'Null_Percentage': round(null_percentage, 2),
            'Non_Null_Count': total_rows - null_count
        })
    
    null_df = pd.DataFrame(null_analysis)
    
    # Save for pipeline integration
    output_path = OUTPUT_DIR / "null_counts.csv"
    null_df.to_csv(output_path, index=False)
    
    # Log critical findings
    high_null_columns = null_df[null_df['Null_Percentage'] > 20]['Column'].tolist()
    if high_null_columns:
        logger.warning(f"High null rate (>20%) in columns: {', '.join(high_null_columns)}")
    
    logger.info(f"Completeness analysis saved to: {output_path}")


def analyze_cardinality(df: pd.DataFrame) -> None:
    """
    Analyze unique value distributions for feature engineering decisions.
    
    Business Purpose:
        - Identifies high-cardinality features requiring encoding
        - Detects potential data quality issues (unexpected duplicates)
        - Guides categorical feature handling strategies
        
    Args:
        df: Dataframe to analyze for cardinality
    """
    logger.info("Analyzing feature cardinality...")
    
    cardinality_analysis = []
    total_rows = len(df)
    
    for column in df.columns:
        unique_count = df[column].nunique()
        unique_percentage = (unique_count / total_rows) * 100
        
        cardinality_analysis.append({
            'Column': column,
            'Unique_Count': unique_count,
            'Unique_Percentage': round(unique_percentage, 2),
            'Total_Count': total_rows
        })
    
    cardinality_df = pd.DataFrame(cardinality_analysis)
    
    # Save for feature engineering reference
    output_path = OUTPUT_DIR / "unique_counts.csv"
    cardinality_df.to_csv(output_path, index=False)
    
    logger.info(f"Cardinality analysis saved to: {output_path}")


def perform_business_sanity_checks(df: pd.DataFrame) -> None:
    """
    Execute business rule validations for data quality assurance.
    
    Business Purpose:
        - Validates data against known business constraints
        - Identifies transactions requiring special handling
        - Provides metrics for data quality dashboards
        
    Business Rules Checked:
        1. Customer ID completeness (required for CLV analysis)
        2. Negative quantities in non-credit transactions
        3. Zero/negative pricing anomalies
        4. Credit note identification and volume
        5. Duplicate transaction detection
        6. Date range validation for fiscal year alignment
        7. Geographic distribution for market analysis
        
    Args:
        df: Transaction dataframe to validate
    """
    logger.info("Performing business rule validation...")
    
    total_rows = len(df)
    
    # Calculate key metrics
    customer_id_missing = df['Customer ID'].isnull().sum()
    customer_id_missing_pct = (customer_id_missing / total_rows) * 100
    
    # Identify credit notes (returns/refunds)
    is_credit_note = df['Invoice'].astype(str).str.startswith('C')
    credit_notes_count = is_credit_note.sum()
    credit_notes_pct = (credit_notes_count / total_rows) * 100
    
    # Check for anomalous quantities in regular sales
    non_credit_mask = ~is_credit_note
    negative_qty_in_sales = (df['Quantity'] <= 0) & non_credit_mask
    negative_qty_count = negative_qty_in_sales.sum()
    negative_qty_pct = (negative_qty_count / total_rows) * 100
    
    # Check for pricing anomalies
    zero_price_in_sales = (df['Price'] <= 0) & non_credit_mask
    zero_price_count = zero_price_in_sales.sum()
    zero_price_pct = (zero_price_count / total_rows) * 100
    
    # Duplicate detection
    duplicate_rows = df.duplicated()
    duplicate_count = duplicate_rows.sum()
    duplicate_pct = (duplicate_count / total_rows) * 100
    
    # Date range validation
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    date_min = df['InvoiceDate'].min()
    date_max = df['InvoiceDate'].max()
    
    # Geographic distribution
    country_distribution = df['Country'].value_counts().head(10)
    total_countries = df['Country'].nunique()
    
    # Generate comprehensive report
    output_path = OUTPUT_DIR / "sanity_checks.md"
    with open(output_path, 'w') as f:
        f.write("# Business Rule Validation Report\n\n")
        
        f.write("## Customer Data Completeness\n")
        f.write(f"- **Customer ID missing:** {customer_id_missing:,} rows ({customer_id_missing_pct:.2f}%)\n")
        f.write("  - *Impact:* Cannot perform customer lifetime value or retention analysis\n\n")
        
        f.write("## Transaction Integrity\n")
        f.write(f"- **Negative quantity (non-credits):** {negative_qty_count:,} rows ({negative_qty_pct:.2f}%)\n")
        f.write("  - *Action:* Review for data entry errors or system issues\n")
        f.write(f"- **Zero/negative price (non-credits):** {zero_price_count:,} rows ({zero_price_pct:.2f}%)\n")
        f.write("  - *Action:* Investigate promotional items or data quality issues\n\n")
        
        f.write("## Returns and Credits\n")
        f.write(f"- **Credit notes (Invoice prefix 'C'):** {credit_notes_count:,} rows ({credit_notes_pct:.2f}%)\n")
        f.write("  - *Insight:* Return rate indicator for product quality analysis\n\n")
        
        f.write("## Data Duplication\n")
        f.write(f"- **Exact duplicate rows:** {duplicate_count:,} rows ({duplicate_pct:.2f}%)\n")
        f.write("  - *Action:* Investigate potential double-posting in source systems\n\n")
        
        f.write("## Temporal Coverage\n")
        f.write(f"- **Date range:** {date_min.strftime('%Y-%m-%d')} to {date_max.strftime('%Y-%m-%d')}\n")
        f.write(f"- **Duration:** {(date_max - date_min).days} days\n\n")
        
        f.write("## Geographic Distribution\n")
        f.write(f"- **Total countries:** {total_countries}\n")
        f.write("- **Top 10 markets by transaction volume:**\n")
        for country, count in country_distribution.items():
            pct = (count / total_rows) * 100
            f.write(f"  - {country}: {count:,} transactions ({pct:.2f}%)\n")
    
    logger.info(f"Business validation report saved to: {output_path}")


def analyze_statistical_distribution(df: pd.DataFrame, column: str, feature_name: str) -> Dict:
    """
    Comprehensive statistical analysis for feature engineering decisions.
    
    Business Purpose:
        - Determines appropriate scaling strategies for ML models
        - Identifies skewness requiring transformation
        - Provides visual evidence for outlier treatment decisions
        
    Technical Approach:
        - Multiple visualization types for different insights
        - Shapiro-Wilk test for normality (sampled for performance)
        - Separate linear and log-scale views for skewed data
        
    Args:
        df: Dataframe containing the feature
        column: Column name to analyze
        feature_name: Human-readable name for reports
        
    Returns:
        Dictionary with test statistics and insights
    """
    logger.info(f"Analyzing distribution for {feature_name}...")
    
    # Data preparation - remove infinite and null values
    clean_data = df[column].replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(clean_data) == 0:
        logger.warning(f"No valid data for {feature_name} distribution analysis")
        return {'shapiro_pvalue': None, 'note': 'No valid data'}
    
    # Create comprehensive visualization grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Distribution Analysis: {feature_name}', fontsize=16)
    
    # Panel 1: Raw histogram for natural distribution
    axes[0, 0].hist(clean_data, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Histogram (Raw Scale)')
    axes[0, 0].set_xlabel(feature_name)
    axes[0, 0].set_ylabel('Frequency')
    
    # Panel 2: Log-scale histogram for heavy-tailed distributions
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
    
    # Panel 3: Boxplot for outlier visualization
    axes[1, 0].boxplot(clean_data)
    axes[1, 0].set_title('Boxplot')
    axes[1, 0].set_ylabel(feature_name)
    
    # Panel 4: Q-Q plot for normality assessment
    if len(clean_data) > 3:  # Statistical minimum for Q-Q plot
        stats.probplot(clean_data, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot (Normal Distribution)')
    else:
        axes[1, 1].text(0.5, 0.5, 'Insufficient data\nfor Q-Q plot', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Q-Q Plot - Insufficient Data')
    
    plt.tight_layout()
    plot_path = PLOTS_DIR / f"{feature_name.lower().replace(' ', '_')}_distribution.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate individual plots for detailed analysis
    _save_individual_distribution_plots(clean_data, feature_name)
    
    # Perform Shapiro-Wilk normality test
    # Sample if dataset too large (test becomes oversensitive)
    test_data = clean_data
    if len(test_data) > SHAPIRO_TEST_MAX_SAMPLES:
        test_data = clean_data.sample(SHAPIRO_TEST_MAX_SAMPLES, random_state=42)
        logger.info(f"Sampled {SHAPIRO_TEST_MAX_SAMPLES} observations for normality test")
    
    if len(test_data) >= 3:
        try:
            shapiro_stat, shapiro_pvalue = stats.shapiro(test_data)
            return {
                'shapiro_pvalue': shapiro_pvalue, 
                'shapiro_stat': shapiro_stat,
                'sample_size': len(test_data)
            }
        except Exception as e:
            logger.error(f"Shapiro test failed for {feature_name}: {e}")
            return {'shapiro_pvalue': None, 'note': f'Test failed: {str(e)}'}
    else:
        return {'shapiro_pvalue': None, 'note': 'Insufficient data for test'}


def _save_individual_distribution_plots(data: pd.Series, feature_name: str) -> None:
    """
    Save individual distribution plots for presentation decks.
    
    Internal helper to generate standalone visualizations.
    """
    # Histogram
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=50, alpha=0.7, edgecolor='black')
    plt.title(f'{feature_name} - Histogram')
    plt.xlabel(feature_name)
    plt.ylabel('Frequency')
    plt.savefig(PLOTS_DIR / f"{feature_name.lower().replace(' ', '_')}_hist.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Boxplot
    plt.figure(figsize=(8, 6))
    plt.boxplot(data)
    plt.title(f'{feature_name} - Boxplot')
    plt.ylabel(feature_name)
    plt.savefig(PLOTS_DIR / f"{feature_name.lower().replace(' ', '_')}_boxplot.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Q-Q plot
    plt.figure(figsize=(8, 6))
    if len(data) > 3:
        stats.probplot(data, dist="norm", plot=plt)
        plt.title(f'{feature_name} - Q-Q Plot')
    plt.savefig(PLOTS_DIR / f"{feature_name.lower().replace(' ', '_')}_qq.png", 
                dpi=300, bbox_inches='tight')
    plt.close()


def execute_distribution_analysis(df: pd.DataFrame) -> None:
    """
    Analyze key business metrics for feature scaling decisions.
    
    Business Purpose:
        - Guides feature transformation strategies
        - Identifies metrics requiring special handling
        - Provides statistical evidence for modeling choices
        
    Metrics Analyzed:
        - Quantity: Order size patterns
        - Unit Price: Pricing distribution
        - Basket Value: Total transaction value
        
    Args:
        df: Transaction dataframe to analyze
    """
    logger.info("Executing distribution analysis for key metrics...")
    
    # Calculate basket-level metrics
    # Business logic: Aggregate to invoice level for true basket analysis
    basket_metrics = df.groupby('Invoice').agg({
        'Quantity': 'sum',
        'Price': 'mean',  # Average price per basket
    }).reset_index()
    basket_metrics['basket_value'] = basket_metrics['Quantity'] * basket_metrics['Price']
    
    # Merge basket value back for row-level analysis
    df_enriched = df.merge(
        basket_metrics[['Invoice', 'basket_value']], 
        on='Invoice', 
        how='left'
    )
    
    # Define business metrics for analysis
    metrics_to_analyze = [
        ('Quantity', 'Quantity'),
        ('Price', 'Unit Price'),
        ('basket_value', 'Basket Value')
    ]
    
    normality_results = []
    
    for column, display_name in metrics_to_analyze:
        result = analyze_statistical_distribution(df_enriched, column, display_name)
        normality_results.append({
            'Variable': display_name,
            'Shapiro_P_Value': result.get('shapiro_pvalue'),
            'Shapiro_Statistic': result.get('shapiro_stat'),
            'Sample_Size': result.get('sample_size', 'N/A'),
            'Notes': result.get('note', '')
        })
    
    # Generate normality test report
    output_path = OUTPUT_DIR / "normality_tests.md"
    with open(output_path, 'w') as f:
        f.write("# Statistical Distribution Analysis\n\n")
        f.write("## Shapiro-Wilk Normality Test Results\n\n")
        f.write("| Variable | P-Value | Statistic | Normal? | Sample Size | Notes |\n")
        f.write("|----------|---------|-----------|---------|-------------|-------|\n")
        
        for result in normality_results:
            p_val = result['Shapiro_P_Value']
            stat = result['Shapiro_Statistic']
            
            if p_val is not None:
                # Business interpretation: p < 0.05 suggests non-normal distribution
                normal_status = "No" if p_val < 0.05 else "Yes"
                p_val_str = f"{p_val:.6f}"
                stat_str = f"{stat:.6f}" if stat is not None else "N/A"
            else:
                normal_status = "N/A"
                p_val_str = "N/A"
                stat_str = "N/A"
            
            f.write(f"| {result['Variable']} | {p_val_str} | {stat_str} | "
                   f"{normal_status} | {result['Sample_Size']} | {result['Notes']} |\n")
        
        f.write("\n## Interpretation Guide\n")
        f.write("- **P-value < 0.05**: Reject normality assumption (consider log transformation)\n")
        f.write("- **Large samples**: Test becomes oversensitive; visual inspection recommended\n")
        f.write("- **Business Impact**: Non-normal distributions may require:\n")
        f.write("  - Log or Box-Cox transformation for linear models\n")
        f.write("  - Robust scaling for outlier-sensitive algorithms\n")
        f.write("  - Tree-based models which handle non-normality naturally\n")
    
    logger.info(f"Distribution analysis report saved to: {output_path}")


def detect_technical_anomalies(df: pd.DataFrame) -> None:
    """
    Identify technical data anomalies requiring business investigation.
    
    Business Purpose:
        - Flags potential data entry errors for correction
        - Identifies test transactions for exclusion
        - Highlights unusual patterns requiring business context
        
    Anomaly Categories:
        1. Extreme quantities (±1000) - potential bulk orders or errors
        2. Extreme prices (>£10,000) - luxury items or data errors
        3. Negative quantities without credit note - system issues
        4. Zero pricing without credit note - promotional items or errors
        5. Test/manual adjustment indicators - non-production data
        
    Args:
        df: Transaction dataframe to analyze
    """
    logger.info("Detecting technical anomalies...")
    
    anomaly_results = []
    
    # Anomaly 1: Extreme quantity values suggesting data issues
    qty_extreme = (df['Quantity'] == EXTREME_QUANTITY_THRESHOLD) | \
                  (df['Quantity'] == -EXTREME_QUANTITY_THRESHOLD)
    anomaly_results.append({
        'condition': f'Quantity == ±{EXTREME_QUANTITY_THRESHOLD:,}',
        'business_impact': 'Potential data entry errors or unusual bulk transactions',
        'count': qty_extreme.sum(),
        'percentage': (qty_extreme.sum() / len(df)) * 100,
        'examples': df[qty_extreme].head(5) if qty_extreme.any() else pd.DataFrame()
    })
    
    # Anomaly 2: Extreme pricing requiring investigation
    price_extreme = df['Price'] > EXTREME_PRICE_THRESHOLD
    anomaly_results.append({
        'condition': f'Price > £{EXTREME_PRICE_THRESHOLD:,}',
        'business_impact': 'Luxury items or potential pricing errors',
        'count': price_extreme.sum(),
        'percentage': (price_extreme.sum() / len(df)) * 100,
        'examples': df[price_extreme].head(5) if price_extreme.any() else pd.DataFrame()
    })
    
    # Anomaly 3: Negative quantities without credit note
    is_credit = df['Invoice'].astype(str).str.startswith('C')
    qty_negative_non_credit = (df['Quantity'] < 0) & (~is_credit)
    anomaly_results.append({
        'condition': 'Negative quantity without credit note',
        'business_impact': 'System error - sales should not have negative quantities',
        'count': qty_negative_non_credit.sum(),
        'percentage': (qty_negative_non_credit.sum() / len(df)) * 100,
        'examples': df[qty_negative_non_credit].head(5) if qty_negative_non_credit.any() else pd.DataFrame()
    })
    
    # Anomaly 4: Zero pricing without credit note
    price_zero_non_credit = (df['Price'] == 0) & (~is_credit)
    anomaly_results.append({
        'condition': 'Zero price without credit note',
        'business_impact': 'Free items, promotions, or data quality issues',
        'count': price_zero_non_credit.sum(),
        'percentage': (price_zero_non_credit.sum() / len(df)) * 100,
        'examples': df[price_zero_non_credit].head(5) if price_zero_non_credit.any() else pd.DataFrame()
    })
    
    # Anomaly 5: Test/manual adjustment indicators
    # Convert to string and handle NaN values properly for pattern matching
    desc_pattern = '|'.join(TECHNICAL_ANOMALY_KEYWORDS)
    technical_anomaly = (df['Description']
                        .astype(str)
                        .fillna('')
                        .str.upper()
                        .str.contains(desc_pattern, na=False))
    anomaly_results.append({
        'condition': f'Description contains: {", ".join(TECHNICAL_ANOMALY_KEYWORDS)}',
        'business_impact': 'Test data or manual adjustments - exclude from analysis',
        'count': technical_anomaly.sum(),
        'percentage': (technical_anomaly.sum() / len(df)) * 100,
        'examples': df[technical_anomaly].head(5) if technical_anomaly.any() else pd.DataFrame()
    })
    
    # Generate comprehensive anomaly report
    output_path = OUTPUT_DIR / "anomaly_flags.md"
    with open(output_path, 'w') as f:
        f.write("# Technical Anomaly Detection Report\n\n")
        f.write("## Executive Summary\n")
        
        total_flagged = sum(result['count'] for result in anomaly_results)
        f.write(f"- **Total anomalous transactions:** {total_flagged:,}\n")
        f.write(f"- **Percentage of dataset:** {(total_flagged / len(df)) * 100:.2f}%\n")
        f.write(f"- **Recommendation:** Review flagged transactions before model training\n\n")
        
        for i, result in enumerate(anomaly_results, 1):
            f.write(f"## {i}. {result['condition']}\n\n")
            f.write(f"**Business Impact:** {result['business_impact']}\n\n")
            f.write(f"**Occurrences:** {result['count']:,} ({result['percentage']:.4f}%)\n\n")
            
            if not result['examples'].empty:
                f.write("**Sample transactions:**\n")
                # Select relevant columns for business review
                display_cols = ['Invoice', 'StockCode', 'Description', 'Quantity', 'Price', 'Customer ID']
                available_cols = [col for col in display_cols if col in result['examples'].columns]
                f.write(result['examples'][available_cols].to_markdown(index=False))
                f.write("\n\n")
            else:
                f.write("**No instances found** ✓\n\n")
    
    logger.info(f"Anomaly detection report saved to: {output_path}")


def analyze_basket_value_distribution(df: pd.DataFrame) -> None:
    """
    Analyze transaction basket values for outlier treatment recommendations.
    
    Business Purpose:
        - Identifies appropriate revenue capping thresholds
        - Prevents model bias from extreme transactions
        - Balances data integrity with business reality
        
    Technical Approach:
        - Focuses on positive quantities (actual sales)
        - Calculates true basket values at invoice level
        - Provides percentile-based recommendations
        
    Args:
        df: Transaction dataframe to analyze
    """
    logger.info("Analyzing basket value distribution...")
    
    # Calculate basket values for positive quantities only
    # Business logic: Exclude returns/credits from revenue analysis
    sales_only = df[df['Quantity'] > 0].copy()
    sales_only['line_value'] = sales_only['Quantity'] * sales_only['Price']
    
    # Aggregate to basket level
    basket_analysis = sales_only.groupby('Invoice').agg({
        'line_value': 'sum',
        'Quantity': 'sum',
        'Customer ID': 'first',
        'Country': 'first',
        'InvoiceDate': 'first'
    }).reset_index()
    
    basket_analysis.rename(columns={'line_value': 'basket_value'}, inplace=True)
    
    # Clean data for statistical analysis
    clean_baskets = basket_analysis['basket_value'].replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(clean_baskets) == 0:
        logger.error("No valid basket values found for analysis")
        return
    
    # Calculate key percentiles for business decisions
    percentile_values = [np.percentile(clean_baskets, p) for p in OUTLIER_PERCENTILES]
    top_5_baskets = clean_baskets.nlargest(5).tolist()
    
    # Generate distribution visualizations
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram for distribution shape
    axes[0].hist(clean_baskets, bins=50, alpha=0.7, edgecolor='black')
    axes[0].set_title('Basket Value Distribution')
    axes[0].set_xlabel('Basket Value (£)')
    axes[0].set_ylabel('Frequency')
    axes[0].axvline(percentile_values[1], color='red', linestyle='--', 
                    label=f'99.5th percentile: £{percentile_values[1]:.2f}')
    axes[0].legend()
    
    # Boxplot for outlier visualization
    axes[1].boxplot(clean_baskets)
    axes[1].set_title('Basket Value Outliers')
    axes[1].set_ylabel('Basket Value (£)')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "basket_value_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate strategic recommendations
    output_path = OUTPUT_DIR / "outlier_summary.md"
    with open(output_path, 'w') as f:
        f.write("# Basket Value Analysis & Outlier Treatment Strategy\n\n")
        
        f.write("## Distribution Statistics\n")
        f.write(f"- **Total baskets analyzed:** {len(clean_baskets):,}\n")
        f.write(f"- **Mean basket value:** £{clean_baskets.mean():.2f}\n")
        f.write(f"- **Median basket value:** £{clean_baskets.median():.2f}\n")
        f.write(f"- **Standard deviation:** £{clean_baskets.std():.2f}\n")
        f.write(f"- **Coefficient of variation:** {(clean_baskets.std() / clean_baskets.mean()):.2f}\n\n")
        
        f.write("## Percentile Analysis\n")
        for p, val in zip(OUTLIER_PERCENTILES, percentile_values):
            f.write(f"- **{p}th percentile:** £{val:,.2f}\n")
        f.write("\n")
        
        f.write("## Extreme Values\n")
        f.write("**Top 5 basket values:**\n")
        for i, val in enumerate(top_5_baskets, 1):
            f.write(f"{i}. £{val:,.2f}\n")
        f.write("\n")
        
        f.write("## Outlier Treatment Recommendations\n\n")
        
        # Calculate impact for each threshold
        for p, val in zip(OUTLIER_PERCENTILES, percentile_values):
            affected_count = (clean_baskets > val).sum()
            affected_pct = (affected_count / len(clean_baskets)) * 100
            
            if p == 99:
                threshold_type = "Conservative"
                business_rationale = "Removes only extreme outliers, preserves high-value customer data"
            elif p == 99.5:
                threshold_type = "Balanced (Recommended)"
                business_rationale = "Optimal trade-off between data integrity and model stability"
            else:  # 99.9
                threshold_type = "Liberal"
                business_rationale = "Minimal intervention, retains nearly all transactions"
            
            f.write(f"### {threshold_type} Approach ({p}th percentile): £{val:,.2f}\n")
            f.write(f"- **Transactions affected:** {affected_count:,} ({affected_pct:.2f}%)\n")
            f.write(f"- **Business rationale:** {business_rationale}\n\n")
        
        f.write("## Implementation Strategy\n")
        f.write("1. **For predictive modeling:** Cap at 99.5th percentile (£{:.2f})\n".format(percentile_values[1]))
        f.write("2. **For descriptive analytics:** Flag but retain all values\n")
        f.write("3. **For financial reporting:** No capping, use actual values\n")
        f.write("4. **Next steps:** Review top baskets for business validity before finalizing\n")
    
    logger.info(f"Basket value analysis saved to: {output_path}")


def main():
    """
    Execute comprehensive EDA pipeline with business-focused insights.
    
    Pipeline Stages:
        1. Data consolidation with integrity verification
        2. Schema documentation for stakeholder reference
        3. Data quality assessment (completeness, validity)
        4. Statistical distribution analysis for feature engineering
        5. Anomaly detection for data cleaning decisions
        6. Basket value analysis for outlier treatment
        
    All outputs designed for business stakeholder consumption and
    downstream integration into feature engineering pipelines.
    """
    try:
        logger.info("=" * 60)
        logger.info("ONLINE RETAIL ANALYTICS - EXTENDED EDA PIPELINE")
        logger.info("=" * 60)
        
        # Stage 1: Data consolidation
        df_combined, metadata = load_and_combine_fiscal_years()
        
        # Stage 2: Documentation generation
        generate_schema_documentation(df_combined)
        
        # Stage 3: Data quality assessment
        analyze_data_completeness(df_combined)
        analyze_cardinality(df_combined)
        perform_business_sanity_checks(df_combined)
        
        # Stage 4: Statistical analysis
        execute_distribution_analysis(df_combined)
        
        # Stage 5: Anomaly detection
        detect_technical_anomalies(df_combined)
        
        # Stage 6: Business metric analysis
        analyze_basket_value_distribution(df_combined)
        
        # Pipeline completion summary
        logger.info("\n" + "=" * 60)
        logger.info("EDA PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Analysis outputs: {OUTPUT_DIR}")
        logger.info(f"Visualization assets: {PLOTS_DIR}")
        logger.info(f"Total transactions processed: {metadata['total_row_count']:,}")
        logger.info(f"Source verification: {metadata['source_file_hash'][:16]}...")
        logger.info("Next steps: Review reports and proceed with feature engineering")
        
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        raise
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()