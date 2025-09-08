"""Tests to validate EDA output files and content quality."""

import pandas as pd
import pytest
from pathlib import Path
import re


# Test configuration
EDA_OUTPUT_DIR = Path("docs/eda")
PLOTS_DIR = EDA_OUTPUT_DIR / "plots"

# Expected output files
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

def read_with_encoding_fallback(filepath):
    encodings = ['utf-8', 'cp1252', 'iso-8859-1']
    for encoding in encodings:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {filepath} with any supported encoding")


class TestEDAOutputFiles:
    """Test that all required output files exist."""
    
    def test_output_directories_exist(self):
        """Test that output directories were created."""
        assert EDA_OUTPUT_DIR.exists(), f"EDA output directory {EDA_OUTPUT_DIR} does not exist"
        assert PLOTS_DIR.exists(), f"Plots directory {PLOTS_DIR} does not exist"
    
    def test_markdown_files_exist(self):
        """Test that all required markdown files exist."""
        for filename in EXPECTED_FILES["markdown"]:
            file_path = EDA_OUTPUT_DIR / filename
            assert file_path.exists(), f"Required markdown file {filename} does not exist"
            assert file_path.stat().st_size > 0, f"Markdown file {filename} is empty"
    
    def test_csv_files_exist(self):
        """Test that all required CSV files exist.""" 
        for filename in EXPECTED_FILES["csv"]:
            file_path = EDA_OUTPUT_DIR / filename
            assert file_path.exists(), f"Required CSV file {filename} does not exist"
            assert file_path.stat().st_size > 0, f"CSV file {filename} is empty"
    
    def test_plot_files_exist(self):
        """Test that all required plot files exist."""
        for filename in EXPECTED_FILES["plots"]:
            file_path = PLOTS_DIR / filename
            assert file_path.exists(), f"Required plot file {filename} does not exist"
            assert file_path.stat().st_size > 0, f"Plot file {filename} is empty"


class TestCSVContent:
    """Test CSV file content quality."""
    
    def test_null_counts_csv_structure(self):
        """Test null_counts.csv has proper structure."""
        df = pd.read_csv(EDA_OUTPUT_DIR / "null_counts.csv")
        
        # Check required columns
        required_cols = ['Column', 'Null_Count', 'Null_Percentage', 'Non_Null_Count']
        for col in required_cols:
            assert col in df.columns, f"Required column {col} missing from null_counts.csv"
        
        # Check no NA values in the CSV itself
        assert not df.isnull().any().any(), "null_counts.csv contains NA values"
        
        # Check data types
        assert df['Null_Count'].dtype in ['int64', 'int32'], "Null_Count should be integer"
        assert df['Null_Percentage'].dtype in ['float64', 'float32'], "Null_Percentage should be float"
        assert df['Non_Null_Count'].dtype in ['int64', 'int32'], "Non_Null_Count should be integer"
        
        # Check percentage values are valid
        assert (df['Null_Percentage'] >= 0).all(), "Null percentages should be >= 0"
        assert (df['Null_Percentage'] <= 100).all(), "Null percentages should be <= 100"
    
    def test_unique_counts_csv_structure(self):
        """Test unique_counts.csv has proper structure."""
        df = pd.read_csv(EDA_OUTPUT_DIR / "unique_counts.csv")
        
        # Check required columns
        required_cols = ['Column', 'Unique_Count', 'Unique_Percentage', 'Total_Count']
        for col in required_cols:
            assert col in df.columns, f"Required column {col} missing from unique_counts.csv"
        
        # Check no NA values in the CSV itself
        assert not df.isnull().any().any(), "unique_counts.csv contains NA values"
        
        # Check data types
        assert df['Unique_Count'].dtype in ['int64', 'int32'], "Unique_Count should be integer"
        assert df['Unique_Percentage'].dtype in ['float64', 'float32'], "Unique_Percentage should be float"
        assert df['Total_Count'].dtype in ['int64', 'int32'], "Total_Count should be integer"
        
        # Check percentage values are valid
        assert (df['Unique_Percentage'] >= 0).all(), "Unique percentages should be >= 0"
        assert (df['Unique_Percentage'] <= 100).all(), "Unique percentages should be <= 100"


class TestMarkdownContent:
    """Test markdown file content quality."""
    
    def test_schema_overview_structure(self):
        """Test schema_overview.md has proper table structure."""
        with open(EDA_OUTPUT_DIR / "schema_overview.md", 'r') as f:
            content = f.read()
        
        # Check for header
        assert "# Schema Overview" in content, "Missing main header"
        
        # Check for table structure (markdown tables have | separators)
        lines = content.split('\n')
        table_lines = [line for line in lines if '|' in line and line.strip()]
        
        assert len(table_lines) >= 3, "Should have at least header, separator, and one data row"
        
        # Check first table line has at least 3 columns
        if table_lines:
            first_row_cols = [col.strip() for col in table_lines[0].split('|') if col.strip()]
            assert len(first_row_cols) >= 3, f"Table should have at least 3 columns, found {len(first_row_cols)}"
        
        # Check for expected column headers
        header_line = table_lines[0] if table_lines else ""
        assert "Column" in header_line, "Missing 'Column' header"
        assert "Data_Type" in header_line or "Data Type" in header_line, "Missing data type header"
        assert "Example" in header_line, "Missing example values header"
    
    def test_sanity_checks_structure(self):
        """Test sanity_checks.md has proper structure."""
        with open(EDA_OUTPUT_DIR / "sanity_checks.md", 'r') as f:
            content = f.read()
        
        # Check for main sections
        assert "# Sanity Checks" in content, "Missing main header"
        assert "Missing Data" in content, "Missing Missing Data section"
        assert "Data Quality Issues" in content, "Missing Data Quality section"
        assert "Credit Notes" in content, "Missing Credit Notes section"
        assert "Duplicates" in content, "Missing Duplicates section"
        assert "Date Range" in content, "Missing Date Range section"
        
        # Check for percentage values in content
        percentage_pattern = r'\d+\.\d+%'
        percentages = re.findall(percentage_pattern, content)
        assert len(percentages) >= 3, f"Should have multiple percentage values, found {len(percentages)}"
    
    def test_normality_tests_structure(self):
        """Test normality_tests.md has proper table structure."""
        with open(EDA_OUTPUT_DIR / "normality_tests.md", 'r') as f:
            content = f.read()
        
        # Check for header
        assert "# Normality Tests" in content, "Missing main header"
        assert "Shapiro-Wilk" in content, "Missing Shapiro-Wilk reference"
        
        # Check for table structure
        lines = content.split('\n')
        table_lines = [line for line in lines if '|' in line and line.strip()]
        
        if table_lines:
            # Check first table line has at least 3 columns
            first_row_cols = [col.strip() for col in table_lines[0].split('|') if col.strip()]
            assert len(first_row_cols) >= 3, f"Normality table should have at least 3 columns, found {len(first_row_cols)}"
            
            # Check for expected headers
            header_line = table_lines[0] if table_lines else ""
            assert "Variable" in header_line, "Missing 'Variable' header"
            assert "P-Value" in header_line or "P_Value" in header_line, "Missing P-Value header"
    
    def test_anomaly_flags_structure(self):
        """Test anomaly_flags.md has proper structure."""
        # TODO: remove below two lines
        # with open(EDA_OUTPUT_DIR / "anomaly_flags.md", 'r', encoding='utf-8') as f:
        #    content = f.read()
        content = read_with_encoding_fallback(EDA_OUTPUT_DIR / "anomaly_flags.md")
        
        # Check for main header
        assert "# Technical Value Anomaly Detection" in content, "Missing main header"
        
        # Check for multiple anomaly sections
        section_pattern = r'## \d+\.'
        sections = re.findall(section_pattern, content)
        assert len(sections) >= 3, f"Should have multiple anomaly sections, found {len(sections)}"
        
        # Check for count and percentage patterns
        count_pattern = r'\*\*Count:\*\* \d+(?:,\d+)* rows'
        percentage_pattern = r'\*\*Percentage:\*\* \d+\.\d+%'
        
        counts = re.findall(count_pattern, content)
        percentages = re.findall(percentage_pattern, content)
        
        assert len(counts) >= 3, f"Should have multiple count entries, found {len(counts)}"
        assert len(percentages) >= 3, f"Should have multiple percentage entries, found {len(percentages)}"
    
    def test_outlier_summary_structure(self):
        """Test outlier_summary.md has proper structure."""
        # TODO: remove below two lines
        # with open(EDA_OUTPUT_DIR / "outlier_summary.md", 'r', encoding='utf-8') as f:
        #    content = f.read()
        content = read_with_encoding_fallback(EDA_OUTPUT_DIR / "outlier_summary.md")
        
        # Check for main sections
        assert "# Basket Value Analysis" in content, "Missing main header"
        assert "Basic Statistics" in content, "Missing Basic Statistics section"
        assert "Percentile Analysis" in content, "Missing Percentile Analysis section"
        assert "Top 5 Highest" in content, "Missing Top 5 section"
        assert "Suggested Capping" in content, "Missing Capping Thresholds section"
        
        # Check for monetary values (£ symbol)
        assert "£" in content, "Missing monetary values (£ symbol)"
        
        # Check for percentile mentions
        assert "99th percentile" in content, "Missing 99th percentile"
        assert "99.5th percentile" in content, "Missing 99.5th percentile"


class TestEDALogicalConsistency:
    """Test logical consistency across EDA outputs."""
    
    def test_csv_files_have_consistent_row_counts(self):
        """Test that CSV files reference the same dataset size."""
        null_df = pd.read_csv(EDA_OUTPUT_DIR / "null_counts.csv")
        unique_df = pd.read_csv(EDA_OUTPUT_DIR / "unique_counts.csv")
        
        # Both should have the same number of columns analyzed
        assert len(null_df) == len(unique_df), "Null counts and unique counts should analyze same number of columns"
        
        # Check that total counts are consistent
        if 'Total_Count' in unique_df.columns:
            total_counts = unique_df['Total_Count'].unique()
            assert len(total_counts) == 1, "All columns should reference the same total dataset size"
            
            # Total count should be reasonable (> 1000 for this dataset)
            total_count = total_counts[0]
            assert total_count > 1000, f"Dataset seems too small: {total_count} rows"
    
    def test_percentages_are_reasonable(self):
        """Test that calculated percentages are mathematically reasonable."""
        null_df = pd.read_csv(EDA_OUTPUT_DIR / "null_counts.csv")
        
        # Check that null percentage calculation is correct
        for _, row in null_df.iterrows():
            expected_pct = (row['Null_Count'] / (row['Null_Count'] + row['Non_Null_Count'])) * 100
            actual_pct = row['Null_Percentage']
            
            # Allow small floating point differences
            assert abs(expected_pct - actual_pct) < 0.01, f"Incorrect percentage calculation for {row['Column']}"


def test_eda_pipeline_completeness():
    """High-level test that EDA pipeline generated comprehensive outputs."""
    
    # Count total files generated
    markdown_files = len([f for f in EDA_OUTPUT_DIR.glob("*.md") if f.is_file()])
    csv_files = len([f for f in EDA_OUTPUT_DIR.glob("*.csv") if f.is_file()])
    plot_files = len([f for f in PLOTS_DIR.glob("*.png") if f.is_file()]) if PLOTS_DIR.exists() else 0
    
    # Should have generated multiple files of each type
    assert markdown_files >= 4, f"Should have at least 4 markdown files, found {markdown_files}"
    assert csv_files >= 2, f"Should have at least 2 CSV files, found {csv_files}"
    assert plot_files >= 8, f"Should have at least 8 plot files, found {plot_files}"
    
    print(f"EDA pipeline generated {markdown_files} markdown files, {csv_files} CSV files, and {plot_files} plot files")


if __name__ == "__main__":
    # Run a quick validation when script is executed directly
    test_eda_pipeline_completeness()
    print("Basic EDA output validation completed successfully!")