---
status: complete
phase: 05-ibis-registration-and-integration
source: [05-01-SUMMARY.md, 05-02-SUMMARY.md, 05-03-SUMMARY.md, 05-04-SUMMARY.md, 05-05-SUMMARY.md]
started: 2026-03-15T06:30:00Z
updated: 2026-03-15T07:00:00Z
---

## Current Test

## Current Test

[testing complete]

## Tests

### 1. Ibis schema validate — valid table passes
expected: Call schema.validate(ibis_table) on a valid ibis table (columns match schema, values in range). Validation should return without raising any exception.
result: pass

### 2. Ibis schema validate — invalid table raises SchemaError
expected: Call schema.validate(ibis_table) on an ibis table with a constraint violation (e.g., a value outside allowed range or wrong dtype). A SchemaError should be raised with a meaningful error message identifying the failing column/check.
result: pass

### 3. register_ibis_backends() activates narwhals backend with UserWarning
expected: After calling register_ibis_backends(), a UserWarning containing "narwhals" is emitted indicating the narwhals backend is now active for ibis. Subsequent ibis validation routes through the narwhals DataFrameSchemaBackend.
result: pass

### 4. Unique column constraint works on ibis table
expected: A DataFrameSchema with a column set to unique=True, when validated against an ibis table containing duplicate values in that column, raises a SchemaError reporting uniqueness violation. Valid tables with no duplicates pass.
result: pass
note: "User flagged report_duplicates behavior may not be fully working — likely future phase scope"

### 5. element_wise check on ibis raises SchemaError with NotImplementedError
expected: Define a schema with an element_wise=True check function and validate an ibis table. A SchemaError should be raised. The error message (or str(error)) should contain "NotImplementedError" or "element_wise" indicating that element-wise checks are not supported on SQL-lazy ibis backends.
result: pass

### 6. failure_cases is a native type after ibis validation
expected: When ibis validation raises a SchemaError, error.failure_cases should be a native frame type — one of pd.DataFrame, pl.DataFrame, or pyarrow.Table. It should NOT be a narwhals wrapper object. isinstance checks against these three types should return True.
result: pass
note: "Actual type is ibis.Table (lazy), which is correct — plan 05-04 preserves lazy ibis.Table so callers can .execute()/.to_pandas() themselves. Test description was too narrow."

### 7. lazy=True validation collects multiple ibis errors without crashing
expected: Call schema.validate(ibis_table, lazy=True) on a table with multiple constraint violations across different columns. A SchemaErrors exception (plural) should be raised. Accessing .failure_cases on the exception should not raise any additional error — the failure cases should be accessible as a native frame.
result: pass

### 8. User-defined (custom) check on ibis table executes correctly
expected: Define a DataFrameSchema with a custom (non-builtin, non-element_wise) check function and validate an ibis table. The check function should receive an IbisData object (the ibis table wrapped appropriately). A valid table should pass; a table that violates the custom check should raise SchemaError.
result: issue
reported: "If using lazy validation, there seems to be an error in _count_failure_cases. Non-lazy custom checks work; lazy=True triggers a crash when len() is called on an ibis.Table."
severity: blocker

## Summary

total: 8
passed: 7
issues: 1
pending: 0
skipped: 0

## Gaps

- truth: "lazy=True custom ibis check validation completes without crashing — SchemaError raised for violations, SchemaErrors raised for lazy multi-error collection"
  status: failed
  reason: "User reported: If using lazy validation, there seems to be an error in _count_failure_cases. Non-lazy custom checks work; lazy=True triggers a crash when len() is called on an ibis.Table."
  severity: blocker
  test: 8
  artifacts: []
  missing: []
