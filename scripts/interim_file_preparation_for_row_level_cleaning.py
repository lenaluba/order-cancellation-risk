"""
Online Retail II Data Combiner
==============================

Combines Excel worksheets from two fiscal years into a single Parquet file
for downstream product analytics. Preserves categorical data integrity.

Business Context:
    - Source: UK-based online retail company transaction data
    - Covers 2009-2011 fiscal years in separate Excel sheets
    - Critical for cohort analysis, customer lifetime value, and product performance metrics

Technical Requirements:
    - Preserve invoice prefixes ("C" for credits) for transaction type analysis
    - Maintain customer ID formats (leading zeros, mixed alphanumeric)
    - Use Snappy compression for optimal BI tool query performance

Usage: python combine_retail_data.py
Output: data/interim/df_raw_combined.parquet

Dependencies: pandas>=1.3.0, pyarrow>=5.0.0
"""

from pathlib import Path
import pandas as pd
import hashlib
import logging
import sys

# Configuration - modify these paths for different environments
PROJECT_ROOT = Path(__file__).parent.parent
EXCEL_PATH = PROJECT_ROOT / "data" / "raw" / "online_retail_II.xlsx"
OUTPUT_PATH = PROJECT_ROOT / "data" / "interim" / "df_raw_combined.parquet"

# Critical: These columns MUST remain as strings to preserve business meaning
# pandas auto-infers numeric types which corrupts categorical data
STRING_COLUMNS = [
    "Invoice",      # Contains "C" prefix for credit transactions - essential for revenue calculations
    "StockCode",    # Mixed format product codes (e.g., "85123A", "DOT") - used in product categorization
    "Description",  # Product descriptions with special characters
    "Country",      # Country names for geographic analysis
    "Customer_ID"   # Mixed format IDs with leading zeros - critical for customer analytics
]

SHEET_NAMES = ["Year 2009-2010", "Year 2010-2011"]

# Logging configuration - INFO level for production monitoring
logging.basicConfig(
    level=logging.INFO,
    format= '%(asctime)s - %(name)s - %(module)s:%(lineno)d - %(funcName)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_file_hash(file_path: Path) -> str:
    """
    Calculate SHA-256 hash for data integrity verification and audit trails.
    
    Essential for:
    - Detecting unauthorized data modifications
    - Compliance auditing (SOX, GDPR data lineage)
    - Debugging data pipeline issues
    
    Args:
        file_path: Path to file for hash calculation
        
    Returns:
        Hexadecimal SHA-256 hash string
    """
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        # Process in chunks to handle large files without memory issues
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def basic_data_checks(df: pd.DataFrame, sheet_name: str) -> None:
    """
    Lightweight sanity checks to catch data corruption early in pipeline.
    
    These are NOT data profiling checks - they're circuit breakers for obvious corruption
    that would waste compute resources in downstream feature engineering.
    
    Args:
        df: DataFrame to validate
        sheet_name: Sheet name for error reporting context
        
    Raises:
        ValueError: Critical data issues that prevent further processing
    """
    # Business rule: Retail transaction data should have substantial volume
    # Low counts suggest data loading failures or file corruption
    if len(df) < 1000:
        logger.warning(f"{sheet_name}: Low row count ({len(df):,}) - possible data loading issue")
    
    # Verify core business columns exist - these drive key product metrics
    required_cols = ["Invoice", "StockCode", "Quantity", "InvoiceDate"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"{sheet_name}: Missing critical columns: {missing_cols}")
    
    # Invoice column drives revenue calculations - cannot be entirely empty
    if df["Invoice"].isna().all():
        raise ValueError(f"{sheet_name}: Invoice column is entirely null - data corruption detected")
    
    # Basic temporal validation - retail data should have reasonable date ranges
    if "InvoiceDate" in df.columns:
        try:
            # Convert to datetime with error handling for malformed dates
            date_col = pd.to_datetime(df["InvoiceDate"], errors="coerce")
            if date_col.isna().all():
                logger.warning(f"{sheet_name}: All dates appear invalid - check source format")
            else:
                year_range = date_col.dt.year.dropna()
                if not year_range.empty:
                    # Flag obviously wrong years (data entry errors, format issues)
                    if year_range.min() < 2000 or year_range.max() > 2025:
                        logger.warning(f"{sheet_name}: Unusual date range {year_range.min()}-{year_range.max()}")
        except Exception as e:
            logger.warning(f"{sheet_name}: Could not validate date column: {e}")
    
    logger.info(f"{sheet_name}: Basic data integrity checks passed")


def load_sheet(sheet_name: str) -> pd.DataFrame:
    """
    Load Excel worksheet with categorical data preservation and validation.
    
    Key design decisions:
    - openpyxl engine: Best performance for .xlsx files >1MB
    - String dtype enforcement: Prevents pandas numeric auto-conversion
    - Basic validation: Catch corruption before expensive downstream processing
    
    Args:
        sheet_name: Worksheet name to load from Excel file
        
    Returns:
        DataFrame with preserved categorical data types
        
    Raises:
        FileNotFoundError: Source Excel file not accessible
        ValueError: Data integrity issues that prevent processing
    """
    try:
        # Force categorical columns to string type - critical for downstream analytics
        dtype_map = {col: "string" for col in STRING_COLUMNS}
        
        df = pd.read_excel(
            EXCEL_PATH,
            sheet_name=sheet_name,
            dtype=dtype_map,
            engine="openpyxl"  # Best performance for large Excel files
        )
        
        # Immediate check for completely empty sheets
        if df.empty:
            raise ValueError(f"Sheet '{sheet_name}' contains no data")
        
        # Run data integrity checks before proceeding with expensive operations
        basic_data_checks(df, sheet_name)
            
        logger.info(f"Successfully loaded {sheet_name}: {len(df):,} rows, {len(df.columns)} columns")
        return df
        
    except FileNotFoundError:
        logger.error(f"Excel file not found: {EXCEL_PATH}")
        raise
    except ValueError as e:
        logger.error(f"Data integrity issue in {sheet_name}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading {sheet_name}: {e}")
        raise


def combine_and_save(df_2009: pd.DataFrame, df_2010: pd.DataFrame) -> None:
    """
    Combine fiscal year datasets and persist as analytics-ready Parquet.
    
    Design choices:
    - source_year column: Enables cohort analysis across fiscal years
    - Snappy compression: Optimal balance of compression ratio and query speed for BI tools
    - ignore_index=True: Clean sequential indexing for downstream analytics
    
    Args:
        df_2009: 2009-2010 fiscal year transaction data
        df_2010: 2010-2011 fiscal year transaction data
        
    Raises:
        MemoryError: Insufficient memory for dataset combination
        OSError: File system issues during write operation
    """
    try:
        # Create copies to avoid modifying original dataframes
        df_2009 = df_2009.copy()
        df_2010 = df_2010.copy()
        
        # Add source tracking for audit trail and cohort analysis
        # This enables year-over-year comparisons and data lineage tracking
        df_2009['source_year'] = "2009-2010"
        df_2010['source_year'] = "2010-2011"
        
        # Combine datasets - pandas handles minor schema differences gracefully
        df_combined = pd.concat([df_2009, df_2010], ignore_index=True)
        
        # Ensure output directory structure exists (follows data engineering conventions)
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as Parquet with Snappy compression
        # Snappy chosen for optimal query performance in downstream BI tools (Tableau, PowerBI)
        df_combined.to_parquet(
            OUTPUT_PATH,
            index=False,           # No need for pandas index in analytical datasets
            compression="snappy",   # Fast decompression for interactive queries
            engine="pyarrow"       # Most stable Parquet implementation
        )
        
        # Log success metrics for monitoring
        file_size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
        logger.info(f"Combined dataset saved: {len(df_combined):,} rows ({len(df_2009):,} + {len(df_2010):,})")
        logger.info(f"Output file: {OUTPUT_PATH} ({file_size_mb:.1f}MB)")
        
    except MemoryError:
        logger.error("Insufficient memory to process datasets - consider processing in chunks")
        raise
    except Exception as e:
        logger.error(f"Failed to combine and save data: {e}")
        raise


def main():
    """
    Execute complete data combination pipeline with monitoring and error handling.
    
    Pipeline stages:
    1. Source file integrity verification (hash calculation)
    2. Load and validate both fiscal year datasets  
    3. Combine with audit trail and save as analytics-ready Parquet
    
    Exit codes:
        0: Successful completion
        1: Processing error or user interruption
    """
    try:
        logger.info("Starting Online Retail II data combination pipeline")
        
        # Calculate source file hash for data governance and audit trail
        # Critical for compliance (SOX) and debugging data issues
        try:
            source_hash = calculate_file_hash(EXCEL_PATH)
            logger.info(f"Source file integrity hash: {source_hash}")
        except Exception as e:
            # Hash calculation failure shouldn't stop processing, but should be logged
            logger.warning(f"Could not calculate file hash (proceeding anyway): {e}")
        
        # Load both fiscal year datasets with validation
        df_2009 = load_sheet(SHEET_NAMES[0])
        df_2010 = load_sheet(SHEET_NAMES[1])
        
        # Schema compatibility check - warn but don't fail
        # pandas concat handles minor differences, strict validation often over-constrains
        if set(df_2009.columns) != set(df_2010.columns):
            logger.warning("Column schema mismatch between fiscal years - pandas will handle alignment")
        
        # Combine datasets and save for downstream analytics
        combine_and_save(df_2009, df_2010)
        
        logger.info("Data combination pipeline completed successfully")
        
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()