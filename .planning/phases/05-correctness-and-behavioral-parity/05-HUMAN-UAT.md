---
status: partial
phase: 05-correctness-and-behavioral-parity
source: [05-VERIFICATION.md]
started: 2026-05-25T18:00:00Z
updated: 2026-05-25T18:00:00Z
---

## Current Test

[awaiting human testing]

## Tests

### 1. CORR-01 strict='filter' PySpark integration test
expected: PANDERA_USE_NARWHALS_BACKEND=True pixi run pyspark-test tests/pyspark/test_pyspark_model.py::test_dataframe_schema_strict -x passes without xfail/xpass; schema.validate(df) returns DataFrame with only columns ['a', 'b'] when strict='filter'
result: [pending]

### 2. CORR-02 pandera.schema accessor PySpark integration test
expected: PANDERA_USE_NARWHALS_BACKEND=True pixi run pyspark-test tests/pyspark/test_pyspark_accessor.py::test_dataframe_add_schema -x passes without xfail/xpass; data.pandera.schema == schema1 after schema1(data)
result: [pending]

### 3. TEST-FIX-01 config dict assertions narwhals mode
expected: PANDERA_USE_NARWHALS_BACKEND=True pixi run pyspark-test tests/pyspark/test_pyspark_config.py::TestPanderaConfig -x passes; all 5 tests pass without xfail; use_narwhals_backend=True in both expected and actual dicts
result: [pending]

### 4. TEST-FIX-01 config dict assertions native mode
expected: PANDERA_USE_NARWHALS_BACKEND=False pixi run pyspark-test tests/pyspark/test_pyspark_config.py::TestPanderaConfig -x passes; no regression to native PySpark path; use_narwhals_backend=False in both dicts
result: [pending]

### 5. Full PySpark suite regression check
expected: PANDERA_USE_NARWHALS_BACKEND=True pixi run pyspark-test tests/pyspark/ shows no unexpected new failures; only the 3 known pre-existing xfails remain (group_by limitation, ValueError, coerce_dtype in test_pyspark_model.py)
result: [pending]

## Summary

total: 5
passed: 0
issues: 0
pending: 5
skipped: 0
blocked: 0

## Gaps
