---
status: testing
phase: 04-container-backend-and-polars-registration
source: [04-01-SUMMARY.md, 04-02-SUMMARY.md, 04-03-SUMMARY.md, 04-04-SUMMARY.md]
started: 2026-03-14T13:50:00Z
updated: 2026-03-14T13:50:00Z
---

## Current Test

[testing complete]

## Tests

### 1. validate() preserves DataFrame type
expected: With narwhals backend activated, schema.validate(pl.DataFrame(...)) returns a pl.DataFrame (not a narwhals wrapper or LazyFrame). Example: pa.DataFrameSchema({"a": pa.Column(int)}).validate(pl.DataFrame({"a": [1,2,3]})) should return a pl.DataFrame.
result: pass

### 2. validate() preserves LazyFrame type
expected: With narwhals backend activated, schema.validate(pl.LazyFrame(...)) returns a pl.LazyFrame. The output type should match the input type exactly.
result: pass

### 3. strict=True rejects extra columns
expected: With narwhals backend, schema.validate(df_with_extra_col, lazy=False) raises SchemaError when the DataFrame has columns not defined in the schema and strict=True. The error message should identify the unexpected column.
result: [pending]

### 4. strict="filter" drops extra columns
expected: With narwhals backend, schema.validate(df_with_extra_col) returns a DataFrame containing only the columns defined in the schema when strict="filter". No error raised.
result: [pending]

### 5. lazy=True collects all errors
expected: With narwhals backend and lazy=True, validating a DataFrame with multiple column violations raises SchemaErrors (plural) containing all errors at once — not just the first failure.
result: [pending]

### 6. failure_cases is native pl.DataFrame
expected: When a check fails, SchemaError.failure_cases is a native pl.DataFrame (not a narwhals wrapper). isinstance(e.failure_cases, pl.DataFrame) should be True.
result: [pending]

### 7. opt-in via config_context
expected: Without any activation, pa.DataFrameSchema(...).validate(pl.DataFrame(...)) uses the default Polars backend. Wrapping with pandera.config_context(use_narwhals_backend=True) routes the same call through the narwhals DataFrameSchemaBackend.
result: [pending]

### 8. opt-in via env var PANDERA_USE_NARWHALS_BACKEND
expected: Setting PANDERA_USE_NARWHALS_BACKEND=True (or =1) in the environment activates the narwhals backend globally, equivalent to config_context(use_narwhals_backend=True).
result: [pending]

## Summary

total: 8
passed: 2
issues: 0
pending: 6
skipped: 0

## Gaps

[none yet]
