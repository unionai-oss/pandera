---
phase: 11-round-4-pr-review-fixes
plan: "01"
subsystem: narwhals-backend
tags:
  - pyspark
  - narwhals
  - schema-errors
  - dead-code-removal
  - tdd
dependency_graph:
  requires:
    - "10-02"
  provides:
    - "SE-01: unified SchemaErrors contract for PySpark Narwhals"
    - "DC-01: dead PySpark branch removed from _materialize()"
  affects:
    - pandera/backends/narwhals/container.py
    - pandera/api/narwhals/utils.py
    - tests/narwhals/test_phase01_arch.py
tech_stack:
  added: []
  patterns:
    - "TDD RED/GREEN cycle per task"
    - "Unified SchemaErrors raise for all Narwhals backends including PySpark"
key_files:
  modified:
    - pandera/backends/narwhals/container.py
    - pandera/api/narwhals/utils.py
    - tests/narwhals/test_phase01_arch.py
decisions:
  - "SE-01: PySpark Narwhals now raises SchemaErrors like Polars/Ibis (no accessor protocol)"
  - "Intentional CORR-02 regression: df.pandera.schema no longer set on narwhals PySpark path (unifies with Polars/Ibis behavior)"
  - "DC-01: _materialize() PySpark branch deleted (previously documented as dead code in Phase 10)"
  - "Old ARCH-04 tests deleted; replaced with SE-01/DC-01 regression guards"
metrics:
  duration: "~20 minutes"
  completed_date: "2026-05-30"
  tasks_completed: 2
  files_modified: 3
---

# Phase 11 Plan 01: SE-01/DC-01 PySpark Narwhals SchemaErrors Alignment Summary

Unified the PySpark Narwhals backend validation contract with Polars/Ibis Narwhals by removing the `is_pyspark` accessor-protocol dispatch from `DataFrameSchemaBackend.validate()`, deleting `_handle_pyspark_validation_result`, and excising the dead `nw.Implementation.PYSPARK` branch from `_materialize()`.

## Tasks Completed

| Task | Description | Commit |
|------|-------------|--------|
| 1 (RED) | Add failing SE-01 regression tests | 05a52362 |
| 1 (GREEN) | Remove is_pyspark branch, _handle_pyspark_validation_result, ARCH-04 tests, NIT-05 hoist | c4cc8879 |
| 2 (RED) | Add failing DC-01 regression test for _materialize | 0ada93a1 |
| 2 (GREEN) | Remove dead PySpark branch from _materialize(); update docstring | 0322e60e |

## What Changed

### pandera/backends/narwhals/container.py

- Deleted the `is_pyspark = check_lf.implementation in (...)` assignment and its comment block (lines 101-111)
- Removed `if is_pyspark:` sub-branch inside `drop_invalid_rows` path (was calling `_handle_pyspark_validation_result`)
- Removed `elif is_pyspark:` error path (was setting `df.pandera.errors` instead of raising)
- Removed `if is_pyspark:` success path (was calling `_handle_pyspark_validation_result` with `has_errors=False`)
- Deleted the entire `_handle_pyspark_validation_result` method (lines 266-303)
- PySpark Narwhals validation now raises `SchemaErrors` via the unified path, matching Polars/Ibis Narwhals

### pandera/api/narwhals/utils.py

- Deleted the `nw.Implementation.PYSPARK` / `PYSPARK_CONNECT` branch from `_materialize()` (the `import pyarrow as pa`, `.first()`, and PyArrow table construction)
- Simplified docstring to document only the two live branches: `nw.LazyFrame.collect()` and SQL-lazy `execute()`
- `_SQL_LAZY_IMPLEMENTATIONS` frozenset unchanged (PYSPARK/PYSPARK_CONNECT entries retained for `_is_sql_lazy`)

### tests/narwhals/test_phase01_arch.py

- Deleted ARCH-04 section (5 tests: `test_handle_pyspark_validation_result_exists`, `_error_path`, `_success_path`, `_has_docstring`, `test_validate_calls_handle_pyspark_validation_result`)
- NIT-05: removed 3 duplicate inner `import inspect` statements (already imported at module top level)
- Added `test_validate_has_no_is_pyspark_branch_after_se01` — SE-01 regression guard
- Added `test_validate_no_handle_pyspark_method_after_se01` — SE-01 regression guard
- Added `test_materialize_has_no_pyspark_branch_after_dc01` — DC-01 regression guard

## Verification Results

- `tests/narwhals/test_phase01_arch.py`: 20/20 passed
- `tests/narwhals/test_e2e.py`: 39 passed, 1 failing (`test_pyspark_builtin_check_passes`) — **expected**, per plan objective
- The 4 PySpark e2e tests that assert `df_out.pandera.errors == {}` now fail because `_handle_pyspark_validation_result` no longer sets the accessor; Plan 02 must fix those

## Intentional Regression Notes

**CORR-02 intentional regression:** Phase 5's CORR-02 fix added `check_obj.pandera.add_schema(schema)` inside `_handle_pyspark_validation_result`. Deleting that method means:
- PySpark Narwhals success/drop-invalid-rows path no longer sets `df.pandera.schema`
- This is intentional: Polars/Ibis Narwhals backends never call `add_schema` either
- Plan 02 must apply `xfail(condition=CONFIG.use_narwhals_backend)` markers or backend-aware helpers on PySpark tests that assert `.pandera.schema` is set

**Plan 02 dependency:** Plan 02 must follow to fix the 4 PySpark e2e tests that still assert `df.pandera.errors` behavior.

## Deviations from Plan

None — plan executed exactly as written.

## Self-Check: PASSED

- `pandera/backends/narwhals/container.py` exists: FOUND
- `pandera/api/narwhals/utils.py` exists: FOUND
- `tests/narwhals/test_phase01_arch.py` exists: FOUND
- Commit 05a52362 exists: FOUND (test RED SE-01)
- Commit c4cc8879 exists: FOUND (feat GREEN SE-01)
- Commit 0ada93a1 exists: FOUND (test RED DC-01)
- Commit 0322e60e exists: FOUND (feat GREEN DC-01)
- `grep -c is_pyspark container.py` (non-comment): 0
- `grep -c _handle_pyspark_validation_result container.py` (non-comment): 0
- `grep -c pyarrow utils.py`: 0
- All 20 arch tests pass
