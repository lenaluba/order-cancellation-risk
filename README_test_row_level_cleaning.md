# Test Coverage for row_level_cleaning.py

This test suite provides comprehensive coverage for all functions in the `row_level_cleaning.py` script.

## Test Organization

The tests are organized into classes, one for each function or logical group:

### Configuration & Setup Tests
- **TestLoadConfig**: Tests YAML configuration loading
- **TestNormalizeColumnNames**: Tests column name normalization
- **TestLoadAndCombineData**: Tests Excel data loading and sheet combination

### Cleaning Step Tests
- **TestStep0NormalizeColumns**: Tests initial column normalization
- **TestStep1CollapseDuplicates**: Tests duplicate removal and aggregation
- **TestStep2DropTechnicalPlaceholders**: Tests removal of test/placeholder data
- **TestStep3DropExtremePrices**: Tests extreme price filtering
- **TestStep4DropZeroPriceNonCredit**: Tests zero price handling
- **TestStep5FixCreditNotePrefix**: Tests credit note prefix correction
- **TestStep6DropNegativeNonCredit**: Tests negative quantity handling
- **TestStep7HandleMissingCustomerId**: Tests missing ID dropping/imputation
- **TestStep8ComputeBasketValue**: Tests basket value calculation
- **TestStep9ApplyCaps**: Tests price and basket value capping
- **TestStep10CreateCountryBucket**: Tests country categorization
- **TestStep10bCreateIsCancelled**: Tests cancelled order indicator
- **TestStep11IsolationForest**: Tests multivariate outlier detection
- **TestStep12Assertions**: Tests data validation assertions

### Utility Function Tests
- **TestCreateWaterfallLog**: Tests waterfall log generation
- **TestCreateCleaningLog**: Tests cleaning log markdown generation
- **TestMainFunction**: Tests main function integration

## Key Test Features

### Mock Data
- Uses pandas DataFrames with representative sample data
- Includes edge cases like credit notes, missing values, extreme prices
- Generates synthetic data for outlier detection tests

### Edge Cases Covered
- Exact duplicates vs. aggregatable duplicates
- Credit notes with 'C' prefix
- Missing customer IDs (both drop and impute scenarios)
- Zero and negative prices
- Extreme values and outliers
- Date range validation
- Mixed data types in columns

### Assertions Tested
- No duplicate invoice-stockcode pairs
- No zero/negative prices in non-credit transactions
- Date range compliance
- Data type consistency

## Running the Tests

```bash
# Run all tests
pytest tests/test_row_level_cleaning.py -v

# Run specific test class
pytest tests/test_row_level_cleaning.py::TestStep7HandleMissingCustomerId -v

# Run with coverage (requires pytest-cov)
pytest tests/test_row_level_cleaning.py --cov=scripts.row_level_cleaning

# Run specific test method
pytest tests/test_row_level_cleaning.py::TestStep7HandleMissingCustomerId::test_impute_missing_id
```

## Test Results Summary
- **Total Tests**: 34
- **All tests passing**: ✅
- **Coverage**: Comprehensive coverage of all public functions
- **Edge cases**: Well-covered including error conditions

## Dependencies
- pytest
- pytest-mock
- pandas
- numpy
- pyyaml
- scikit-learn (for IsolationForest)