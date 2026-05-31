---
status: complete
phase: 06-eliminate-unnecessary-materialization-lazy-first-failure-cases-and-check-output
source: [06-01-SUMMARY.md, 06-02-SUMMARY.md, 06-03-SUMMARY.md]
started: 2026-03-23T22:30:00Z
updated: 2026-03-23T22:45:00Z
---

## Current Test

[testing complete]

## Tests

### 1. Polars subsample() stays lazy
expected: Run `pytest tests/backends/narwhals/test_components.py::TestSubsample -v` — all 4 polars subsample tests pass (head, tail, head+tail, no-params)
result: pass

### 2. Ibis subsample tail= raises NotImplementedError
expected: Run `pytest tests/backends/narwhals/test_components.py::TestSubsample::test_subsample_ibis_tail_raises -v` — test passes, confirming ibis tail= raises NotImplementedError (matching the element_wise pattern)
result: pass

### 3. Polars SchemaError.failure_cases is native pl.DataFrame
expected: Run `pytest tests/backends/narwhals/test_e2e.py -k "failure_cases" -v` — polars tests assert `isinstance(err.failure_cases, pl.DataFrame)` and pass. No narwhals wrapper around failure_cases.
result: pass

### 4. Ibis SchemaError.failure_cases is native ibis.Table
expected: Run `pytest tests/backends/narwhals/test_e2e.py -k "ibis" -v` — ibis tests assert `isinstance(err.failure_cases, ibis.Table)` (not pl.DataFrame, not nw.DataFrame) and pass. Validation does not force Arrow roundtrip conversion.
result: pass
note: pre-existing failure test_custom_check_receives_table_and_key ('Table' == 'DatabaseTable' ibis API rename) — deferred since Plan 01, not introduced by Phase 6

### 5. failure_cases_metadata returns ibis.Table for ibis inputs
expected: Run `pytest tests/backends/narwhals/test_components.py::test_failure_cases_metadata_ibis_returns_ibis_table -v` — test passes confirming FailureCaseMetadata.failure_cases is ibis.Table (not pl.DataFrame).
result: pass

### 6. Full narwhals backend suite passes (no regressions)
expected: Run `pytest tests/backends/narwhals/ -v --ignore=tests/backends/narwhals/test_e2e.py` then `pytest tests/backends/narwhals/test_e2e.py -v` — all tests pass except the 1 pre-existing failure: `test_custom_check_receives_table_and_key` (ibis API DatabaseTable→Table rename, pre-existing before Phase 6).
result: pass

## Summary

total: 6
passed: 6
issues: 0
pending: 0
skipped: 0

## Gaps

[none yet]
