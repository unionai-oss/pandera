---
phase: 06-test-coverage-and-minor-fixes
plan: 01
subsystem: tests/narwhals
tags:
  - pyspark
  - narwhals
  - e2e
  - test-coverage
dependency_graph:
  requires:
    - pandera.backends.pyspark.register (register_pyspark_backends)
    - pandera.backends.narwhals.checks (NarwhalsCheckBackend)
    - pandera.backends.narwhals.container (_handle_pyspark_validation_result)
  provides:
    - tests/narwhals/test_e2e.py PySpark section (TEST-E2E-01)
    - tests/narwhals/conftest.py pyspark backend registration
  affects:
    - narwhals nox session (pyspark matrix entry runs these tests)
tech_stack:
  added:
    - pyspark.sql.SparkSession (module-scoped fixture in e2e test)
    - packaging.version (pyspark version guard for Java 17 workaround)
  patterns:
    - try/except ImportError guard for optional pyspark dependency
    - accessor-based error contract: df_out.pandera.errors (not pytest.raises)
    - DataFrameSchema(unique="x") for schema-level uniqueness (not Column-level)
key_files:
  modified:
    - tests/narwhals/conftest.py
    - tests/narwhals/test_e2e.py
decisions:
  - PySpark Column API does not expose unique=True; used DataFrameSchema(unique="x") instead
  - spark fixture is module-scoped with stop() teardown; pyspark_df is function-scoped
  - _spark_env_vars autouse fixture is function-scoped and no-ops when pyspark absent
metrics:
  duration: 3m 27s
  completed: "2026-05-25T23:52:39Z"
  tasks_completed: 2
  tasks_total: 2
  files_modified: 2
---

# Phase 6 Plan 1: PySpark E2E Tests for Narwhals Backend Summary

**One-liner:** PySpark e2e section added to narwhals test_e2e.py covering backend registration, return-type preservation, passing/failing builtin checks with accessor-based error inspection, nullable=False, and schema-level unique constraints.

## What Was Built

Added a PySpark section to `tests/narwhals/test_e2e.py` that exercises the narwhals backend end-to-end for `pyspark.sql.DataFrame` inputs, mirroring the existing polars/ibis sections. Also extended `tests/narwhals/conftest.py` to re-register pyspark backends when pyspark is importable.

### Task 1: Extend conftest.py (commit: 070c838b)

Added a `try/except ImportError` block in `_ensure_narwhals_backends_registered` that imports `register_pyspark_backends` from `pandera.backends.pyspark.register`, calls `cache_clear()` then `register_pyspark_backends()`. The guard ensures polars-only and ibis-only narwhals environments are unaffected.

### Task 2: Add PySpark section to test_e2e.py (commit: cbdd9dab)

Added:
- `import os` at module level
- `try: import pyspark.sql... HAS_PYSPARK = True; except ImportError: HAS_PYSPARK = False` import block
- `pyspark_only = pytest.mark.skipif(not HAS_PYSPARK, ...)` skip marker
- `_spark_env_vars` autouse function-scoped fixture (sets SPARK_LOCAL_IP, PYARROW_IGNORE_TIMEZONE)
- `spark` module-scoped fixture with Java 17 workaround for pyspark >= 4.0
- `pyspark_df` function-scoped fixture producing a `pyspark.sql.DataFrame` with int columns x, y
- 6 tests decorated with `@pyspark_only`:
  - `test_narwhals_backend_registered_for_pyspark_dataframe`
  - `test_pyspark_dataframe_returns_pyspark_dataframe`
  - `test_pyspark_builtin_check_passes`
  - `test_pyspark_builtin_check_fails`
  - `test_pyspark_nullable_false_fails`
  - `test_pyspark_unique_constraint_fails`

## Verification Results

- 43 tests pass (37 existing + 6 new pyspark tests) with `PANDERA_USE_NARWHALS_BACKEND=True`
- `grep -c "^@pyspark_only" tests/narwhals/test_e2e.py` = 7 (6 tests + spark fixture) >= 6
- `grep -c "df_out.pandera.errors" tests/narwhals/test_e2e.py` = 4 >= 4
- All tests use accessor-based error contract (`df_out.pandera.errors`), not `pytest.raises(SchemaError)`

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] PySpark Column API does not support unique=True**
- **Found during:** Task 2 execution — `TypeError: Column.__init__() got an unexpected keyword argument 'unique'`
- **Issue:** The plan specified `pa_pyspark.Column("int", unique=True)` but PySpark's Column class does not expose a `unique` keyword argument; uniqueness is a schema-level constraint in the PySpark API
- **Fix:** Changed to `DataFrameSchema({"x": pa_pyspark.Column("int")}, unique="x")` which matches the actual PySpark API pattern (verified in `tests/pyspark/test_pyspark_container.py`)
- **Files modified:** `tests/narwhals/test_e2e.py`
- **Commit:** cbdd9dab

## Known Stubs

None — all tests assert on real behavior with real data.

## Threat Flags

None — test-only changes with no new network endpoints, auth paths, or schema changes.

## Self-Check: PASSED

- [x] tests/narwhals/conftest.py exists and contains pyspark re-registration
- [x] tests/narwhals/test_e2e.py exists and contains 6 pyspark tests
- [x] Commit 070c838b exists: feat(06-01): extend narwhals conftest to re-register pyspark backends
- [x] Commit cbdd9dab exists: feat(06-01): add PySpark e2e section to tests/narwhals/test_e2e.py
- [x] 43 tests pass, 0 failures, 0 regressions
