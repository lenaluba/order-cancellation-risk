"""
EDA Output Validation Test Suite
================================

Business Purpose:
    - Ensures EDA pipeline produces complete, consistent outputs
    - Validates data quality metrics meet business requirements
    - Confirms reports are stakeholder-ready with proper formatting

Technical Approach:
    - File existence and completeness validation
    - Content structure verification
    - Cross-file consistency checks
    - Mathematical accuracy validation

Dependencies: pytest>=6.0, pandas>=1.3.0
"""

import sys
import os
import pandas as pd
import pytest
from pathlib import Path
import re

# Module import pattern for robust test execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

# Test configuration
EDA_OUTPUT_DIR = Path("docs/eda")
PLOTS_DIR = EDA_OUTPUT_DIR / "plots"

# Expected output files - aligned with enterprise EDA pipeline
EXPECTED_FILES = {
    "markdown": [
        "schema_overview.md",
        "sanity_checks.md", 
        "normality_tests.md",
        "anomaly_flags.md",
        "outlier_summary.md"
    ],
    "csv": [
        "null_counts.csv",
        "unique_counts.csv"
    ],
    "plots": [
        "quantity_distribution.png",
        "quantity_hist.png", 
        "quantity_boxplot.png",
        "quantity_qq.png",
        "unit_price_distribution.png",
        "unit_price_hist.png",
        "unit_price_boxplot.png", 
        "unit_price_qq.png",
        "basket_value_distribution.png",
        "basket_value_hist.png",
        "basket_value_boxplot.png",
        "basket_value_qq.png",
        "basket_value_analysis.png"
    ]
}


def read_with_encoding_fallback(filepath: Path) -> str:
    """
    Read file with multiple encoding attempts for cross-platform compatibility.
    
    Handles various encodings to ensure tests work across different environments.
    """
    encodings = ['utf-8', 'cp1252', 'iso-8859-1']
    for encoding in encodings:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {filepath} with any supported encoding")


class TestEDAOutputFiles:
    """Validate all required output files exist with non-zero content."""
    
    def test_output_directories_exist(self):
        """Verify output directory structure was created."""
        assert EDA_OUTPUT_DIR.exists(), f"EDA output directory {EDA_OUTPUT_DIR} does not exist"
        assert PLOTS_DIR.exists(), f"Plots directory {PLOTS_DIR} does not exist"
    
    def test_markdown_files_exist(self):
        """Verify all business reports were generated."""
        for filename in EXPECTED_FILES["markdown"]:
            file_path = EDA_OUTPUT_DIR / filename
            assert file_path.exists(), f"Required markdown file {filename} does not exist"
            assert file_path.stat().st_size > 0, f"Markdown file {filename} is empty"
    
    def test_csv_files_exist(self):
        """Verify data quality metrics were exported.""" 
        for filename in EXPECTED_FILES["csv"]:
            file_path = EDA_OUTPUT_DIR / filename
            assert file_path.exists(), f"Required CSV file {filename} does not exist"
            assert file_path.stat().st_size > 0, f"CSV file {filename} is empty"
    
    def test_plot_files_exist(self):
        """Verify all statistical visualizations were created."""
        for filename in EXPECTED_FILES["plots"]:
            file_path = PLOTS_DIR / filename
            assert file_path.exists(), f"Required plot file {filename} does not exist"
            assert file_path.stat().st_size > 0, f"Plot file {filename} is empty"


class TestCSVContent:
    """Validate CSV content structure and data integrity."""
    
    def test_null_counts_csv_structure(self):
        """Validate completeness analysis output format."""
        df = pd.read_csv(EDA_OUTPUT_DIR / "null_counts.csv")
        
        # Verify required columns
        required_cols = ['Column', 'Null_Count', 'Null_Percentage', 'Non_Null_Count']
        for col in required_cols:
            assert col in df.columns, f"Required column {col} missing from null_counts.csv"
        
        # Verify data integrity
        assert not df.isnull().any().any(), "null_counts.csv contains NA values"
        
        # Verify data types
        assert df['Null_Count'].dtype in ['int64', 'int32'], "Null_Count should be integer"
        assert df['Null_Percentage'].dtype in ['float64', 'float32'], "Null_Percentage should be float"
        assert df['Non_Null_Count'].dtype in ['int64', 'int32'], "Non_Null_Count should be integer"
        
        # Verify business constraints
        assert (df['Null_Percentage'] >= 0).all(), "Null percentages should be >= 0"
        assert (df['Null_Percentage'] <= 100).all(), "Null percentages should be <= 100"
    
    def test_unique_counts_csv_structure(self):
        """Validate cardinality analysis output format."""
        df = pd.read_csv(EDA_OUTPUT_DIR / "unique_counts.csv")
        
        # Verify required columns
        required_cols = ['Column', 'Unique_Count', 'Unique_Percentage', 'Total_Count']
        for col in required_cols:
            assert col in df.columns, f"Required column {col} missing from unique_counts.csv"
        
        # Verify data integrity
        assert not df.isnull().any().any(), "unique_counts.csv contains NA values"
        
        # Verify data types
        assert df['Unique_Count'].dtype in ['int64', 'int32'], "Unique_Count should be integer"
        assert df['Unique_Percentage'].dtype in ['float64', 'float32'], "Unique_Percentage should be float"
        assert df['Total_Count'].dtype in ['int64', 'int32'], "Total_Count should be integer"
        
        # Verify business constraints
        assert (df['Unique_Percentage'] >= 0).all(), "Unique percentages should be >= 0"
        assert (df['Unique_Percentage'] <= 100).all(), "Unique percentages should be <= 100"


class TestMarkdownContent:
    """Validate markdown report content and structure."""
    
    def test_schema_overview_structure(self):
        """Validate data dictionary format and content."""
        content = read_with_encoding_fallback(EDA_OUTPUT_DIR / "schema_overview.md")
        
        # Verify main header
        assert "# Schema Overview" in content, "Missing main header"
        
        # Verify table structure (markdown tables have | separators)
        lines = content.split('\n')
        table_lines = [line for line in lines if '|' in line and line.strip()]
        
        assert len(table_lines) >= 3, "Should have at least header, separator, and one data row"
        
        # Verify column structure
        if table_lines:
            first_row_cols = [col.strip() for col in table_lines[0].split('|') if col.strip()]
            assert len(first_row_cols) >= 3, f"Table should have at least 3 columns, found {len(first_row_cols)}"
        
        # Verify expected column headers
        header_line = table_lines[0] if table_lines else ""
        assert "Column" in header_line, "Missing 'Column' header"
        assert "Data_Type" in header_line or "Data Type" in header_line, "Missing data type header"
        assert "Example" in header_line, "Missing example values header"
    
    def test_sanity_checks_structure(self):
        """Validate business rule validation report structure."""
        content = read_with_encoding_fallback(EDA_OUTPUT_DIR / "sanity_checks.md")
        
        # Check for updated header (enterprise version)
        assert ("# Business Rule Validation Report" in content or 
                "# Sanity Checks" in content), "Missing main header"
        
        # Check for key business validation sections - flexible matching
        required_sections = [
            (r"Customer\s*(ID|Data)", "Customer data validation"),
            (r"Transaction\s*(Integrity|Quality)", "Transaction integrity checks"),
            (r"Credit\s*(Notes?|Returns?)", "Credit note analysis"),
            (r"Duplicat", "Duplicate detection"),
            (r"Date\s*(Range|Coverage)", "Date range validation"),
            (r"Geographic", "Geographic analysis")
        ]
        
        for section_pattern, description in required_sections:
            if not re.search(section_pattern, content, re.IGNORECASE):
                # Fallback to simpler keyword search
                simple_keyword = section_pattern.split(r'\s*')[0].replace(r'(', '').replace(r')', '')
                assert simple_keyword in content, f"Missing {description} section"
        
        # Verify percentage values present
        percentage_pattern = r'\d+\.\d+%'
        percentages = re.findall(percentage_pattern, content)
        assert len(percentages) >= 3, f"Should have multiple percentage values, found {len(percentages)}"
    
    def test_normality_tests_structure(self):
        """Validate statistical distribution analysis report."""
        content = read_with_encoding_fallback(EDA_OUTPUT_DIR / "normality_tests.md")
        
        # Check for updated header
        assert ("# Statistical Distribution Analysis" in content or
                "# Normality Tests" in content), "Missing main header"
        assert "Shapiro-Wilk" in content, "Missing Shapiro-Wilk reference"
        
        # Verify table structure
        lines = content.split('\n')
        table_lines = [line for line in lines if '|' in line and line.strip()]
        
        if table_lines:
            # Verify column count
            first_row_cols = [col.strip() for col in table_lines[0].split('|') if col.strip()]
            assert len(first_row_cols) >= 3, f"Normality table should have at least 3 columns, found {len(first_row_cols)}"
            
            # Verify expected headers
            header_line = table_lines[0] if table_lines else ""
            assert "Variable" in header_line, "Missing 'Variable' header"
            assert "P-Value" in header_line or "P_Value" in header_line, "Missing P-Value header"
    
    def test_anomaly_flags_structure(self):
        """Validate technical anomaly detection report."""
        content = read_with_encoding_fallback(EDA_OUTPUT_DIR / "anomaly_flags.md")
        
        # Check for updated header
        assert ("# Technical Anomaly Detection Report" in content or
                "# Technical Value Anomaly Detection" in content), "Missing main header"
        
        # Verify multiple anomaly sections
        section_pattern = r'## \d+\.'
        sections = re.findall(section_pattern, content)
        assert len(sections) >= 3, f"Should have multiple anomaly sections, found {len(sections)}"
        
        # Verify business impact documentation (enterprise addition)
        if "Business Impact:" in content:
            impact_pattern = r'\*\*Business Impact:\*\*'
            impacts = re.findall(impact_pattern, content)
            assert len(impacts) >= 3, "Each anomaly should document business impact"
        
        # Verify count and percentage patterns - flexible for both formats
        # Original: **Count:** 123 rows
        # Enterprise: **Occurrences:** 123 (0.01%)
        count_patterns = [
            r'\*\*Count:\*\*\s*[\d,]+\s*rows',
            r'\*\*Occurrences:\*\*\s*[\d,]+\s*\(',
            r'Count:\s*[\d,]+',
            r'Occurrences:\s*[\d,]+'
        ]
        
        count_found = False
        for pattern in count_patterns:
            if re.search(pattern, content):
                count_found = True
                break
        
        assert count_found, "Should have count/occurrence information in anomaly report"
        
        # Verify percentage values
        percentage_pattern = r'\d+\.\d+%'
        percentages = re.findall(percentage_pattern, content)
        assert len(percentages) >= 3, f"Should have multiple percentage entries, found {len(percentages)}"
    
    def test_outlier_summary_structure(self):
        """Validate basket value analysis report."""
        content = read_with_encoding_fallback(EDA_OUTPUT_DIR / "outlier_summary.md")
        
        # Check for updated headers
        assert ("# Basket Value Analysis" in content or
                "# Basket Value Analysis & Outlier Treatment Strategy" in content), "Missing main header"
        
        # Verify required sections - flexible matching for both versions
        required_sections = [
            ("Statistics", "Distribution statistics"),
            ("Percentile", "Percentile analysis"),
            ("Top 5", "Extreme values"),
            ("Recommendation", "Treatment recommendations"),
            ("Capping|Treatment|Implementation", "Outlier handling strategy")  # Flexible matching
        ]
        
        for section_keyword, description in required_sections:
            # Use regex for OR patterns
            if "|" in section_keyword:
                pattern = rf"({section_keyword})"
                assert re.search(pattern, content, re.IGNORECASE), f"Missing {description} section"
            else:
                assert section_keyword in content, f"Missing {description} section"
        
        # Verify monetary values
        assert "£" in content, "Missing monetary values (£ symbol)"
        
        # Verify percentile mentions
        assert "99th percentile" in content, "Missing 99th percentile"
        assert "99.5th percentile" in content, "Missing 99.5th percentile"


class TestEDALogicalConsistency:
    """Validate cross-file consistency and mathematical accuracy."""
    
    def test_csv_files_have_consistent_row_counts(self):
        """Verify both CSV files reference same dataset."""
        null_df = pd.read_csv(EDA_OUTPUT_DIR / "null_counts.csv")
        unique_df = pd.read_csv(EDA_OUTPUT_DIR / "unique_counts.csv")
        
        # Both should analyze same number of columns
        assert len(null_df) == len(unique_df), \
            "Null counts and unique counts should analyze same number of columns"
        
        # Verify consistent total counts
        if 'Total_Count' in unique_df.columns:
            total_counts = unique_df['Total_Count'].unique()
            assert len(total_counts) == 1, "All columns should reference same dataset size"
            
            # Verify reasonable dataset size (business constraint)
            total_count = total_counts[0]
            assert total_count > 1000, f"Dataset seems too small: {total_count} rows"
    
    def test_percentages_are_mathematically_correct(self):
        """Verify percentage calculations are accurate."""
        null_df = pd.read_csv(EDA_OUTPUT_DIR / "null_counts.csv")
        
        # Verify null percentage calculation
        for _, row in null_df.iterrows():
            total = row['Null_Count'] + row['Non_Null_Count']
            if total > 0:  # Avoid division by zero
                expected_pct = (row['Null_Count'] / total) * 100
                actual_pct = row['Null_Percentage']
                
                # Allow small floating point differences
                assert abs(expected_pct - actual_pct) < 0.01, \
                    f"Incorrect percentage calculation for {row['Column']}"


def test_eda_pipeline_completeness():
    """Validate comprehensive output generation from EDA pipeline."""
    
    # Count generated files by type
    markdown_files = len([f for f in EDA_OUTPUT_DIR.glob("*.md") if f.is_file()])
    csv_files = len([f for f in EDA_OUTPUT_DIR.glob("*.csv") if f.is_file()])
    plot_files = len([f for f in PLOTS_DIR.glob("*.png") if f.is_file()]) if PLOTS_DIR.exists() else 0
    
    # Verify minimum output requirements
    assert markdown_files >= 4, f"Should have at least 4 markdown reports, found {markdown_files}"
    assert csv_files >= 2, f"Should have at least 2 CSV exports, found {csv_files}"
    assert plot_files >= 8, f"Should have at least 8 visualizations, found {plot_files}"
    
    print(f"✓ EDA pipeline generated {markdown_files} reports, {csv_files} data files, and {plot_files} visualizations")


if __name__ == "__main__":
    # Quick validation for direct execution
    test_eda_pipeline_completeness()
    print("✓ Basic EDA output validation completed successfully!")