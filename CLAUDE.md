# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build/Test Commands
- **Install dependencies**: `pip install -r requirements.txt`
- **Download dataset**: `make download` or `python scripts/data_ingestion.py`
- **Run EDA pipeline**: `python scripts/eda_extended.py` (creates `docs/eda/` outputs)
- **Run full pipeline**: `make run` or `python main.py`
- **Run tests**: `make test` or `pytest`
- **Test EDA outputs**: `pytest tests/test_eda_outputs.py` (validates EDA files exist and have correct structure)
- **Clean outputs**: `make clean`

## Project Architecture
This is an order cancellation risk prediction project using the Online Retail II dataset from UCI ML Repository.

### Pipeline Flow
1. **Data Ingestion** (`scripts/data_ingestion.py`): Downloads and validates UCI dataset zip file, extracts Excel file to `data/raw/`
2. **EDA Pipeline** (`scripts/eda_extended.py`): Comprehensive exploratory data analysis with outputs to `docs/eda/`
3. **Feature Engineering** (`scripts/feature_engineering.py`): Creates predictive features from raw transaction data
4. **Model Training** (`scripts/model_training.py`): Trains LightGBM model with Optuna hyperparameter optimization
5. **Main Pipeline** (`main.py`): Orchestrates the full end-to-end workflow

### Key Components
- **Target Variable**: `full_basket_return` (indicates order cancellation/return risk)
- **Data Privacy**: CustomerID hashing and timestamp truncation required for GDPR compliance
- **Model Stack**: LightGBM for classification, SHAP for explainability, Optuna for tuning
- **Validation**: SHA-256 hash verification for dataset integrity

### Data Flow
- Raw data: `data/raw/online_retail_II.xlsx`
- Processed features: generated in-memory, passed between pipeline stages
- Model outputs: `outputs/model.pkl`
- Evaluation artifacts: `outputs/` directory (metrics, plots, SHAP values)

### Dependencies
- Core ML: pandas, scikit-learn, lightgbm
- Optimization: optuna for hyperparameter tuning
- Explainability: shap for model interpretation
- Data processing: polars for performance-critical operations
- EDA: matplotlib, seaborn, scipy for statistical analysis and visualization

## Dataset Schema
**IMPORTANT**: The Online Retail II dataset uses different column names than standard retail datasets:
- `Invoice` (not `InvoiceNo`)
- `Customer ID` (not `CustomerID`) 
- `Price` (not `UnitPrice`)
- `InvoiceDate` (standard)
- `Description`, `Quantity`, `Country` (standard)

## Environment
Environment is defined in environment.yml. It uses Python 3.10.

## Common Issues & Solutions

### EDA Pipeline
- **Tests fail before running EDA**: Run `python scripts/eda_extended.py` first to generate outputs before testing
- **Column name errors**: Dataset uses `Invoice`, `Customer ID`, `Price` (with spaces) - not standard retail column names
- **Missing scipy**: EDA requires `scipy` for statistical tests (Shapiro-Wilk normality tests)
- **Plot generation**: Uses matplotlib/seaborn, saves to `docs/eda/plots/` directory
- **File encoding**: Use `encoding='utf-8'` when reading generated markdown files to avoid Windows encoding issues

### Data Quality Patterns
- Credit notes: Invoices starting with "C" indicate returns/cancellations
- Negative quantities: Should only appear in credit note transactions
- Missing Customer ID: High percentage (~25%) of transactions lack customer identification
- Price anomalies: Some transactions have extreme unit prices (>£10,000) requiring investigation
- Geographic concentration: UK dominates transaction volume (~90%+)

## Code Style Guidelines**
- **Linting/Type Checking**: Apply linting and type checking after code changes to ensure quality
- **Testing**: All new functionality requires corresponding pytest tests
- **Error Handling**: Use try/except with specific exceptions, log error details
- **Logging**: Use structured logging with module/function context
- **Modularisation**: Single responsibility per function (max 50 lines preferred). Clear separation: data loading, transformation, validation, output Configuration isolated from logic (use constants/config sections).

## **Code Documentation**
Code should read like a technical document that a business analyst could understand the purpose of, while maintaining the rigor expected in production data pipelines at scale.
**Module-level docstrings**:  Include business context, data lineage, and downstream impact
**Function docstrings**: Google-style docstrings for functions and classes - Specify business purpose, not just technical implementation
**Inline comments**: Explain "why" decisions were made, not "what" the code does. Code Comments Should Answer:
 - Why was this approach chosen over alternatives?
 - What business rules drive this logic?
 - What are the downstream dependencies?
 - What edge cases are we handling/ignoring and why?
- **Additional code and documentation instructions**

## **Additional Coding and Code Documentation Guidelines**:
Read `./docs/coding_and_code_documentation_standards.md` for **detailed coding and code documentation instructions**.

## Git Workflow
- **Never commit unless explicitly asked**: User prefers to review changes before committing
- **Commit messages**: Should be concise and describe the "why" not just the "what"
- **Pre-commit hooks**: May modify files automatically, requiring commit amendment

## Performance Considerations
- **Large Dataset**: Online Retail II has ~1M+ transactions, consider memory usage
- **Batch Processing**: Process data in chunks when dealing with full dataset
- **Model Training**: Optuna optimization can be time-consuming, consider n_trials parameter

## Testing Best Practices
- **EDA Output Tests**: Always run EDA pipeline before testing outputs
- **File Encoding**: Use encoding fallback ['utf-8', 'cp1252', 'iso-8859-1'] when reading files on Windows for compatibility across environments
- **Test Data**: Consider using subset of data for faster test execution
- **Mock External APIs**: Mock data downloads in unit tests to avoid network dependencies


### Creating Tests for Scripts Directory
When creating tests for modules in `scripts/` directory, follow these patterns to avoid common import failures:

**Module Import Pattern:**
```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

try:
    from module_name import function1, function2
except ImportError:
    pytest.skip("module_name not available", allow_module_level=True)
```

**Patching Best Practices:**
- Use `patch.object(module, 'CONSTANT', value)` instead of `patch('scripts.module.CONSTANT', value)`
- Import the module locally within test methods to avoid module-level import errors
- Use `patch.object(module, 'function_name')` for better isolation

**Dependency Management:**
- Check all transitive dependencies are installed (scipy, matplotlib, seaborn, pyarrow)
- Use conditional imports with pytest.skip for optional dependencies
- Mock heavy dependencies (matplotlib.pyplot) when testing logic, not visualization

**Hash Testing:**
- Use `hashlib.sha256("content".encode()).hexdigest()` for dynamic hash calculation
- Don't hardcode expected hashes - calculate them dynamically for cross-platform compatibility

**File Path Handling:**
- Use `tmp_path` fixture for temporary files in tests
- Use `patch.object()` to mock file paths rather than string-based patching
- Test both file existence and non-existence scenarios