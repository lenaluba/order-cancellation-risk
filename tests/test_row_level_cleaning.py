"""
Test suite for row_level_cleaning.py functions.
Tests all cleaning steps with mock data.
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, mock_open
import yaml

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

try:
    from row_level_cleaning import (
        load_config, normalize_column_names, load_and_combine_data,
        step_0_normalize_columns, step_1_collapse_duplicates,
        step_2_drop_technical_placeholders, step_3_drop_extreme_prices,
        step_4_drop_zero_price_non_credit, step_5_fix_credit_note_prefix,
        step_6_drop_negative_non_credit, step_7_handle_missing_customer_id,
        step_8_compute_basket_value, step_9_apply_caps,
        step_10_create_country_bucket, step_10b_create_is_cancelled,
        step_11_isolation_forest, step_12_assertions,
        create_waterfall_log, create_cleaning_log, EU_COUNTRIES
    )
except ImportError:
    pytest.skip("row_level_cleaning module not available", allow_module_level=True)


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'DROP_MISSING_ID': True,
        'PRICE_CAP_Q': 0.99,
        'BASKET_CAP_Q': 0.995,
        'ISOLATION_CONTAM': 0.001
    }


@pytest.fixture
def sample_dataframe():
    """Create a sample dataframe for testing."""
    base_date = datetime(2010, 6, 1)
    return pd.DataFrame({
        'Invoice': ['489434', '489435', 'C489436', '489437', '489438'],
        'StockCode': ['21232', '21233', '21234', '21235', '21236'],
        'Description': ['Product A', 'TEST PRODUCT', 'Product C', 'Product D', 'Product E'],
        'Quantity': [2, 1000, -1, 3, 0],
        'InvoiceDate': [base_date + timedelta(days=i) for i in range(5)],
        'Price': [10.5, 20.0, 15.0, 10000.5, 0.0],
        'Customer ID': [12345.0, 12346.0, np.nan, 12348.0, 12349.0],
        'Country': ['United Kingdom', 'France', 'Germany', 'United States', 'United Kingdom']
    })


@pytest.fixture
def normalized_dataframe():
    """Create a normalized column names dataframe."""
    base_date = datetime(2010, 6, 1)
    return pd.DataFrame({
        'invoice': ['489434', '489435', 'C489436', '489437', '489438'],
        'stockcode': ['21232', '21233', '21234', '21235', '21236'],
        'description': ['Product A', 'TEST PRODUCT', 'Product C', 'Product D', 'Product E'],
        'quantity': [2, 1000, -1, 3, 0],
        'invoicedate': [base_date + timedelta(days=i) for i in range(5)],
        'price': [10.5, 20.0, 15.0, 10000.5, 0.0],
        'customer_id': [12345.0, 12346.0, np.nan, 12348.0, 12349.0],
        'country': ['United Kingdom', 'France', 'Germany', 'United States', 'United Kingdom']
    })


class TestLoadConfig:
    """Test configuration loading."""
    
    def test_load_config_success(self, tmp_path, sample_config):
        """Test successful config loading."""
        config_path = tmp_path / "test_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(sample_config, f)
        
        loaded_config = load_config(config_path)
        assert loaded_config == sample_config
    
    def test_load_config_file_not_found(self):
        """Test config loading with non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_config(Path("non_existent_config.yaml"))


class TestNormalizeColumnNames:
    """Test column name normalization."""
    
    def test_normalize_column_names(self, sample_dataframe):
        """Test column name normalization."""
        df = normalize_column_names(sample_dataframe.copy())
        expected_columns = ['invoice', 'stockcode', 'description', 'quantity', 
                          'invoicedate', 'price', 'customer_id', 'country']
        assert list(df.columns) == expected_columns
    
    def test_normalize_already_normalized(self, normalized_dataframe):
        """Test normalizing already normalized columns."""
        df = normalize_column_names(normalized_dataframe.copy())
        assert list(df.columns) == list(normalized_dataframe.columns)


class TestLoadAndCombineData:
    """Test data loading and combining."""
    
    @patch('row_level_cleaning.pd.read_excel')
    def test_load_and_combine_data(self, mock_read_excel, sample_dataframe):
        """Test loading and combining Excel sheets."""
        # Mock two sheets with different data
        df1 = sample_dataframe.iloc[:3].copy()
        df2 = sample_dataframe.iloc[3:].copy()
        
        mock_read_excel.side_effect = [df1, df2]
        
        result = load_and_combine_data(Path("test.xlsx"))
        
        assert len(result) == len(sample_dataframe)
        assert mock_read_excel.call_count == 2
        # Check that Invoice and StockCode are strings
        assert result['Invoice'].dtype == 'object'
        assert result['StockCode'].dtype == 'object'


class TestStep0NormalizeColumns:
    """Test step 0 normalization."""
    
    def test_step_0_normalize(self, sample_dataframe):
        """Test step 0 column normalization."""
        df = step_0_normalize_columns(sample_dataframe.copy())
        assert 'invoice' in df.columns
        assert 'customer_id' in df.columns
        assert 'Customer ID' not in df.columns


class TestStep1CollapseDuplicates:
    """Test duplicate collapsing."""
    
    def test_collapse_exact_duplicates(self, normalized_dataframe):
        """Test removing exact duplicates."""
        # Add exact duplicate
        df = pd.concat([normalized_dataframe, normalized_dataframe.iloc[[0]]], 
                       ignore_index=True)
        
        result, removed = step_1_collapse_duplicates(df)
        assert len(result) == len(normalized_dataframe)
        assert removed == 1
    
    def test_collapse_aggregate_quantities(self, normalized_dataframe):
        """Test aggregating same invoice-stockcode pairs."""
        # Add row with same invoice-stockcode but different quantity
        duplicate_row = normalized_dataframe.iloc[[0]].copy()
        duplicate_row['quantity'] = 5
        df = pd.concat([normalized_dataframe, duplicate_row], ignore_index=True)
        
        result, removed = step_1_collapse_duplicates(df)
        
        # Check that quantities were summed
        aggregated_row = result[(result['invoice'] == '489434') & 
                               (result['stockcode'] == '21232')]
        assert len(aggregated_row) == 1
        assert aggregated_row['quantity'].iloc[0] == 7  # 2 + 5


class TestStep2DropTechnicalPlaceholders:
    """Test technical placeholder removal."""
    
    def test_drop_quantity_1000(self, normalized_dataframe):
        """Test dropping rows with quantity = 1000."""
        result, removed = step_2_drop_technical_placeholders(normalized_dataframe)
        assert removed == 1
        assert 1000 not in result['quantity'].values
    
    def test_drop_test_description(self, normalized_dataframe):
        """Test dropping rows with TEST in description."""
        result, removed = step_2_drop_technical_placeholders(normalized_dataframe)
        assert removed == 1
        assert not result['description'].str.contains('TEST', case=False).any()
    
    def test_drop_manual_adjust_description(self, normalized_dataframe):
        """Test dropping rows with MANUAL or ADJUST in description."""
        df = normalized_dataframe.copy()
        df.loc[0, 'description'] = 'MANUAL ADJUSTMENT'
        
        result, removed = step_2_drop_technical_placeholders(df)
        assert not result['description'].str.contains('MANUAL', case=False).any()
        assert not result['description'].str.contains('ADJUST', case=False).any()


class TestStep3DropExtremePrices:
    """Test extreme price removal."""
    
    def test_drop_extreme_prices(self, normalized_dataframe):
        """Test dropping rows with price > 10000."""
        result, removed = step_3_drop_extreme_prices(normalized_dataframe)
        assert removed == 1
        assert result['price'].max() <= 10000


class TestStep4DropZeroPriceNonCredit:
    """Test zero price non-credit removal."""
    
    def test_drop_zero_price_non_credit(self, normalized_dataframe):
        """Test dropping rows with price=0 and non-credit invoice."""
        result, removed = step_4_drop_zero_price_non_credit(normalized_dataframe)
        assert removed == 1
        
        # Check that credit notes with zero price are kept
        credit_zero = result[(result['invoice'].str.startswith('C')) & 
                            (result['price'] == 0)]
        assert len(credit_zero) == 0  # None in our test data
    
    def test_keep_zero_price_credit(self, normalized_dataframe):
        """Test keeping rows with price=0 for credit notes."""
        df = normalized_dataframe.copy()
        df.loc[4, 'invoice'] = 'C489438'  # Make it a credit note
        
        result, removed = step_4_drop_zero_price_non_credit(df)
        assert removed == 0  # Should keep the zero-price credit note


class TestStep5FixCreditNotePrefix:
    """Test credit note prefix fixing."""
    
    def test_fix_all_negative_invoice(self):
        """Test adding C prefix to invoices with all negative quantities."""
        df = pd.DataFrame({
            'invoice': ['489999', '489999', '490000'],
            'quantity': [-1, -2, 3]
        })
        
        result = step_5_fix_credit_note_prefix(df)
        
        # Invoice 489999 should have C prefix
        assert result[result['invoice'] == 'C489999']['quantity'].tolist() == [-1, -2]
        # Invoice 490000 should remain unchanged
        assert '490000' in result['invoice'].values
    
    def test_no_fix_needed(self, normalized_dataframe):
        """Test when no invoices need fixing."""
        result = step_5_fix_credit_note_prefix(normalized_dataframe)
        # No changes should be made
        assert result['invoice'].tolist() == normalized_dataframe['invoice'].tolist()


class TestStep6DropNegativeNonCredit:
    """Test negative quantity non-credit removal."""
    
    def test_drop_negative_non_credit(self):
        """Test dropping negative quantities in non-credit invoices."""
        df = pd.DataFrame({
            'invoice': ['489999', 'C490000', '490001'],
            'quantity': [-1, -2, 3]
        })
        
        result, removed = step_6_drop_negative_non_credit(df)
        assert removed == 1
        assert len(result) == 2
        
        # Credit note with negative should be kept
        assert len(result[result['invoice'] == 'C490000']) == 1


class TestStep7HandleMissingCustomerId:
    """Test missing customer ID handling."""
    
    def test_drop_missing_id(self, normalized_dataframe, sample_config):
        """Test dropping rows with missing customer ID."""
        result, removed = step_7_handle_missing_customer_id(
            normalized_dataframe, sample_config
        )
        assert removed == 1
        assert not result['customer_id'].isna().any()
    
    def test_impute_missing_id(self, normalized_dataframe):
        """Test imputing missing customer IDs."""
        config = {'DROP_MISSING_ID': False}
        result, removed = step_7_handle_missing_customer_id(
            normalized_dataframe, config
        )
        
        assert removed == 0
        assert 'missing_id' in result.columns
        
        # Check imputed value
        imputed_rows = result[result['missing_id'] == 1]
        assert len(imputed_rows) == 1
        assert imputed_rows['customer_id'].iloc[0] == 'anon_germany'


class TestStep8ComputeBasketValue:
    """Test basket value computation."""
    
    def test_compute_basket_value(self, normalized_dataframe):
        """Test computing basket value for positive quantities."""
        result = step_8_compute_basket_value(normalized_dataframe)
        
        assert 'basket_value' in result.columns
        
        # Check positive quantity calculation
        positive_mask = result['quantity'] > 0
        expected_values = result.loc[positive_mask, 'quantity'] * result.loc[positive_mask, 'price']
        assert result.loc[positive_mask, 'basket_value'].tolist() == expected_values.tolist()
        
        # Check negative/zero quantities have 0 basket value
        non_positive_mask = result['quantity'] <= 0
        assert (result.loc[non_positive_mask, 'basket_value'] == 0).all()


class TestStep9ApplyCaps:
    """Test price and basket value capping."""
    
    def test_apply_caps(self, sample_config):
        """Test applying price and basket caps."""
        df = pd.DataFrame({
            'invoice': ['489999'] * 100,
            'price': np.linspace(1, 100, 100),
            'basket_value': np.linspace(10, 1000, 100)
        })
        
        result, caps = step_9_apply_caps(df, sample_config)
        
        # Check caps were calculated
        assert 'price_cap' in caps
        assert 'basket_cap' in caps
        
        # Check values are capped
        assert result['price'].max() <= caps['price_cap']
        assert result['basket_value'].max() <= caps['basket_cap']
    
    def test_caps_exclude_credit_notes(self, sample_config):
        """Test that caps are calculated excluding credit notes."""
        df = pd.DataFrame({
            'invoice': ['C489999'] * 50 + ['490000'] * 50,
            'price': np.concatenate([np.ones(50) * 1000, np.linspace(1, 100, 50)]),
            'basket_value': np.concatenate([np.ones(50) * 5000, np.linspace(10, 500, 50)])
        })
        
        result, caps = step_9_apply_caps(df, sample_config)
        
        # Caps should be based on non-credit rows only
        assert caps['price_cap'] < 1000
        assert caps['basket_cap'] < 5000


class TestStep10CreateCountryBucket:
    """Test country bucketing."""
    
    def test_create_country_bucket(self, normalized_dataframe):
        """Test creating country buckets."""
        result = step_10_create_country_bucket(normalized_dataframe)
        
        assert 'country_bucket' in result.columns
        
        # Check specific mappings
        uk_rows = result[result['country'] == 'United Kingdom']
        assert (uk_rows['country_bucket'] == 'UK').all()
        
        france_rows = result[result['country'] == 'France']
        assert (france_rows['country_bucket'] == 'EU').all()
        
        us_rows = result[result['country'] == 'United States']
        assert (us_rows['country_bucket'] == 'NonEU').all()
    
    def test_unknown_country(self):
        """Test handling of missing country values."""
        df = pd.DataFrame({'country': [None, np.nan, 'United Kingdom']})
        result = step_10_create_country_bucket(df)
        
        assert result.loc[0, 'country_bucket'] == 'Unknown'
        assert result.loc[1, 'country_bucket'] == 'Unknown'


class TestStep10bCreateIsCancelled:
    """Test is_cancelled indicator creation."""
    
    def test_create_is_cancelled(self, normalized_dataframe):
        """Test creating is_cancelled indicator."""
        result = step_10b_create_is_cancelled(normalized_dataframe)
        
        assert 'is_cancelled' in result.columns
        
        # Check credit notes are marked as 1
        credit_rows = result[result['invoice'].str.startswith('C')]
        assert (credit_rows['is_cancelled'] == 1).all()
        
        # Check non-credit notes are marked as 0
        non_credit_rows = result[~result['invoice'].str.startswith('C')]
        assert (non_credit_rows['is_cancelled'] == 0).all()


class TestStep11IsolationForest:
    """Test multivariate outlier detection."""
    
    def test_isolation_forest_with_contamination(self, sample_config):
        """Test isolation forest outlier removal."""
        # Create data with clear outliers
        np.random.seed(42)
        n_samples = 1000
        df = pd.DataFrame({
            'invoice': ['490000'] * n_samples,
            'quantity': np.random.normal(10, 2, n_samples),
            'price': np.random.normal(20, 5, n_samples),
            'basket_value': np.random.normal(200, 50, n_samples)
        })
        
        # Add some outliers
        df.loc[0:5, 'price'] = 1000
        df.loc[0:5, 'quantity'] = 1000
        
        result, removed = step_11_isolation_forest(df, sample_config)
        
        assert removed > 0
        assert len(result) < len(df)
    
    def test_isolation_forest_zero_contamination(self):
        """Test isolation forest with zero contamination."""
        df = pd.DataFrame({
            'invoice': ['490000'] * 10,
            'quantity': [1] * 10,
            'price': [10] * 10,
            'basket_value': [10] * 10
        })
        
        config = {'ISOLATION_CONTAM': 0}
        result, removed = step_11_isolation_forest(df, config)
        
        assert removed == 0
        assert len(result) == len(df)


class TestStep12Assertions:
    """Test data validation assertions."""
    
    def test_valid_data_passes(self):
        """Test that valid data passes all assertions."""
        base_date = datetime(2010, 6, 1)
        df = pd.DataFrame({
            'invoice': ['489999', '490000', 'C490001'],
            'stockcode': ['A', 'B', 'C'],
            'price': [10, 20, 30],
            'invoicedate': [base_date] * 3
        })
        
        # Should not raise any assertions
        step_12_assertions(df)
    
    def test_duplicate_invoice_stockcode_fails(self):
        """Test that duplicate invoice-stockcode pairs fail."""
        df = pd.DataFrame({
            'invoice': ['489999', '489999'],
            'stockcode': ['A', 'A'],
            'price': [10, 20],
            'invoicedate': [datetime(2010, 1, 1)] * 2
        })
        
        with pytest.raises(AssertionError, match="duplicate"):
            step_12_assertions(df)
    
    def test_zero_price_non_credit_fails(self):
        """Test that zero price on non-credit rows fails."""
        df = pd.DataFrame({
            'invoice': ['489999'],
            'stockcode': ['A'],
            'price': [0],
            'invoicedate': [datetime(2010, 1, 1)]
        })
        
        with pytest.raises(AssertionError, match="price <= 0"):
            step_12_assertions(df)
    
    def test_date_out_of_range_fails(self):
        """Test that dates outside range fail."""
        df = pd.DataFrame({
            'invoice': ['489999'],
            'stockcode': ['A'],
            'price': [10],
            'invoicedate': [datetime(2008, 1, 1)]  # Too early
        })
        
        with pytest.raises(AssertionError, match="date"):
            step_12_assertions(df)


class TestCreateWaterfallLog:
    """Test waterfall log creation."""
    
    def test_create_waterfall_log(self):
        """Test creating waterfall log dataframe."""
        steps = [
            {'step': 'initial', 'rows_remaining': 1000, 'rows_removed': 0},
            {'step': 'duplicates', 'rows_remaining': 950, 'rows_removed': 50},
            {'step': 'technical', 'rows_remaining': 940, 'rows_removed': 10}
        ]
        
        result = create_waterfall_log(steps)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert list(result.columns) == ['step', 'rows_remaining', 'rows_removed']


class TestCreateCleaningLog:
    """Test cleaning log creation."""
    
    def test_create_cleaning_log(self, sample_config):
        """Test creating cleaning log markdown."""
        df_initial = pd.DataFrame({'col1': range(1000)})
        df_final = pd.DataFrame({
            'col1': range(500),
            'col2': range(500)
        })
        
        steps_info = {
            'step_1': 100,
            'step_2': 50,
            'step_3': 20
        }
        
        caps = {
            'price_cap': 99.99,
            'basket_cap': 999.99,
            'price_capped_count': 10,
            'basket_capped_count': 5
        }
        
        log = create_cleaning_log(df_initial, df_final, steps_info, caps, sample_config)
        
        assert isinstance(log, str)
        assert '# Row-Level Cleaning Log' in log
        assert 'Initial rows: 1,000' in log
        assert 'Final rows: 500' in log
        assert 'Retention rate: 50.0%' in log


class TestMainFunction:
    """Test main function integration."""
    
    @patch('row_level_cleaning.load_and_combine_data')
    @patch('row_level_cleaning.load_config')
    @patch('builtins.open', new_callable=mock_open)
    @patch('row_level_cleaning.Path.exists')
    def test_main_function_integration(self, mock_exists, mock_file, 
                                     mock_load_config, mock_load_data,
                                     sample_config, normalized_dataframe):
        """Test main function with mocked dependencies."""
        # Setup mocks
        mock_exists.return_value = False  # No interim file exists
        mock_load_config.return_value = sample_config
        mock_load_data.return_value = normalized_dataframe.copy()
        
        # Import and patch main to avoid actual file operations
        from row_level_cleaning import main
        
        with patch('row_level_cleaning.pd.DataFrame.to_parquet'):
            with patch('row_level_cleaning.pd.DataFrame.to_csv'):
                with patch('row_level_cleaning.yaml.dump'):
                    with patch('row_level_cleaning.print'):
                        # Run main - should complete without errors
                        main()
        
        # Verify config was loaded
        assert mock_load_config.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])