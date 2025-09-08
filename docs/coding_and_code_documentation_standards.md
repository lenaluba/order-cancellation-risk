## Prompt for Claude Code: Enterprise-Grade Data Science Code Standards

### Overview
Generate production-ready code following best practices from leading tech companies' data science teams, balancing technical excellence with business clarity.

### Core Principles

#### 1. **Documentation Architecture**
- **Module-level docstrings**: Include business context, data lineage, and downstream impact
- **Function docstrings**: Specify business purpose, not just technical implementation
- **Inline comments**: Explain "why" decisions were made, not "what" the code does

#### 2. **Code Structure Requirements**

**Modularization**:
- Single responsibility per function (max 50 lines preferred)
- Clear separation: data loading, transformation, validation, output
- Configuration isolated from logic (use constants/config sections)

**Naming Conventions**:
- Business-meaningful names (e.g., `calculate_customer_lifetime_value` not `calc_clv`)
- Include units in variable names where relevant (`revenue_usd`, `duration_days`)
- Categorical constants in CAPS (e.g., `CREDIT_TRANSACTION_PREFIX = "C"`)

#### 3. **Error Handling Philosophy**
Value-driven balanced approach to error handling, balancing practical efficiency with risk-based prioritisation, focusing on core reliability.

**Layered Approach**:
- **Circuit breakers**: Early validation to prevent expensive downstream failures
- **Specific exceptions**: Business-context exceptions (e.g., `DataQualityError`, `SchemaValidationError`)
- **Graceful degradation**: Log warnings for non-critical issues, fail fast for critical ones

**Focus on Core Reliability**:
- **Essential error handling**: File access, memory issues, write failures
- **Data integrity**: Preserve the dtype enforcement (critical for downstream analytics)
- **Operational visibility**: Keep logging for debugging production issues

**Logging Levels**:
```python
# INFO: Pipeline milestones and metrics
# WARNING: Data quality issues that don't stop processing  
# ERROR: Failures requiring intervention
```
**Logging format**: '%(asctime)s - %(name)s - %(module)s:%(lineno)d - %(funcName)s - %(levelname)s - %(message)s' 

#### 4. **Data Integrity Practices**

**Validation Hierarchy**:
1. **Existence checks**: Required columns/files present
2. **Type preservation**: Maintain categorical data integrity (no auto-conversion)
3. **Business rule validation**: Row counts, date ranges, value distributions
4. **Audit trail**: Hash calculations, source tracking, timestamp logging

**Example Pattern**:
```python
def validate_transaction_data(df: pd.DataFrame) -> None:
    """
    Validate retail transaction data meets business requirements.
    
    Business Rules:
        - Minimum 1000 transactions (indicates successful load)
        - Invoice dates within fiscal year boundaries
        - No negative quantities for regular sales
    """
    # Implementation with specific error messages
```

#### 5. **Performance Considerations**

**Document Trade-offs**:
- Compression choices (e.g., "Snappy for BI tool compatibility vs. gzip for storage")
- Memory management (chunk processing for large datasets)
- Engine selection rationale (e.g., "pyarrow for Parquet stability")

### Deliverable Characteristics

**Each script should include**:
- Business context header explaining the analytical purpose
- Dependency versions for reproducibility
- Clear output specifications (format, location, schema)
- Exit codes with business meaning
- Resource requirements (memory, compute) for large operations

**Code Comments Should Answer**:
- Why was this approach chosen over alternatives?
- What business rules drive this logic?
- What are the downstream dependencies?
- What edge cases are we handling/ignoring and why?

### Anti-Patterns to Avoid
- Magic numbers without context
- Generic variable names (`df`, `data`, `temp`)
- Overly clever one-liners sacrificing readability
- Missing business context in error messages
- Validation without actionable logging

### Appendix - **Module-level docstrings** Structured Template
```
"""
[Component Name] - [Business Purpose]
=====================================

Business Context:
    - Problem being solved
    - Stakeholders/consumers
    - Critical business metrics affected

Technical Approach:
    - Key design decisions
    - Performance optimizations
    - Known limitations

Usage: [execution instructions]
Output: [what gets produced and where]
"""
```

### Key Differentiator
Code should read like a technical document that a business analyst could understand the purpose of, while maintaining the rigor expected in production data pipelines at scale.