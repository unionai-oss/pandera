---
phase: 01-pyspark-registration
plan: "01"
subsystem: backend-registration
tags:
  - pyspark
  - narwhals
  - backend-registration
dependency_graph:
  requires:
    - pandera/backends/narwhals/checks.py
    - pandera/backends/narwhals/components.py
    - pandera/backends/narwhals/container.py
    - pandera/backends/narwhals/builtin_checks.py
    - pandera/config.py (CONFIG.use_narwhals_backend)
  provides:
    - Conditional narwhals/native registration for pyspark_sql.DataFrame
    - register_pyspark_backends() narwhals branch (REG-01)
  affects:
    - pandera/api/pyspark/types.py (bug fix: grpcio-status import guard)
    - tests/narwhals/test_container.py (4 new pyspark registration tests)
tech_stack:
  added: []
  patterns:
    - "if CONFIG.use_narwhals_backend: ... else: ..." conditional registration (mirrors ibis/polars pattern)
    - BACKEND_REGISTRY save/restore in tests for isolation between registration tests
    - try/except ImportError guard for optional Spark Connect dependencies
key_files:
  created: []
  modified:
    - pandera/backends/pyspark/register.py
    - pandera/api/pyspark/types.py
    - tests/narwhals/test_container.py
decisions:
  - "Use check_type= kwarg in get_backend() for pyspark tests — pyspark_sql.DataFrame is passed as a class (not instance) since creating a DataFrame requires SparkSession"
  - "Wrap pyspark.sql.connect import in try/except at module level — grpcio-status may be absent even when pyspark >= 3.4 is installed"
  - "BACKEND_REGISTRY save/restore pattern in tests to achieve isolation — register_backend() uses first-registration-wins semantics, so cache_clear() alone is insufficient"
metrics:
  duration: "~5 minutes"
  completed: "2026-05-10"
  tasks_completed: 2
  files_modified: 3
---

# Phase 01 Plan 01: PySpark Narwhals Registration Summary

Wire PySpark into the Narwhals backend via conditional `if CONFIG.use_narwhals_backend:` branch in `register_pyspark_backends()`, mirroring the ibis/polars registration pattern, with 4 new tests covering activation, native fallback, connect variant, and idempotency.

## What Was Implemented

### Task 1: Failing tests for pyspark narwhals registration

Added 4 test functions to `tests/narwhals/test_container.py` after the existing ibis tests:

1. `test_pyspark_narwhals_activated_when_opted_in` — verifies narwhals backend is registered when `CONFIG.use_narwhals_backend=True`
2. `test_pyspark_native_unchanged_when_flag_off` — verifies native backend is registered when flag is `False`
3. `test_pyspark_connect_narwhals_activated_when_opted_in` — verifies pyspark_connect.DataFrame registration (skipped when grpcio-status not installed)
4. `test_pyspark_register_is_idempotent` — verifies lru_cache prevents duplicate registrations

### Task 2: Conditional narwhals/native branch in register_pyspark_backends()

Modified `pandera/backends/pyspark/register.py` to add:

- `if CONFIG.use_narwhals_backend:` branch importing and registering `NarwhalsCheckBackend`, `ColumnBackend`, `DataFrameSchemaBackend` from `pandera.backends.narwhals.*`
- `import pandera.backends.narwhals.builtin_checks` side-effect import for check dispatcher registration
- `Check.register_backend(nw.LazyFrame, NarwhalsCheckBackend)` per ibis precedent
- Conditional `if PYSPARK_CONNECT_AVAILABLE:` block in narwhals branch for pyspark_connect.DataFrame
- `else:` branch preserving existing native registration verbatim (with aliased imports to avoid shadowing)
- `try/except ImportError` for narwhals import in the narwhals branch (T-01-02 mitigation)

## What Was Tested

All 4 new test functions in `tests/narwhals/test_container.py`:

```
tests/narwhals/test_container.py::test_pyspark_narwhals_activated_when_opted_in PASSED
tests/narwhals/test_container.py::test_pyspark_native_unchanged_when_flag_off PASSED
tests/narwhals/test_container.py::test_pyspark_connect_narwhals_activated_when_opted_in SKIPPED (grpcio-status not installed)
tests/narwhals/test_container.py::test_pyspark_register_is_idempotent PASSED
```

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed module-level pyspark.sql.connect import crash when grpcio-status absent**
- **Found during:** Task 1 test run — ALL 4 pyspark tests failed with `PySparkImportError: grpcio-status >= 1.48.1 must be installed`
- **Issue:** `pandera/backends/pyspark/register.py` line 14 unconditionally imports `from pyspark.sql.connect import dataframe as pyspark_connect` when pyspark >= 3.4, but `pyspark.sql.connect.dataframe` calls `check_dependencies()` which raises `PySparkImportError` if `grpcio-status` is not installed
- **Fix:** Wrapped the module-level import in `try/except Exception: PYSPARK_CONNECT_AVAILABLE = False` so the module can be imported even without Spark Connect dependencies
- **Files modified:** `pandera/backends/pyspark/register.py`

**2. [Rule 1 - Bug] Fixed same module-level import crash in pandera/api/pyspark/types.py**
- **Found during:** Task 2 test run — after fixing register.py, importing `pandera.api.pyspark.container` still crashed due to the same uncaught import in types.py at line 18
- **Issue:** `pandera/api/pyspark/types.py` has the identical uncaught `from pyspark.sql.connect.dataframe import DataFrame as PySparkConnectDataFrame` pattern
- **Fix:** Wrapped in `try/except Exception:` falling back to `from pyspark.sql import DataFrame as PySparkConnectDataFrame` so the module can import even without Spark Connect dependencies
- **Files modified:** `pandera/api/pyspark/types.py`

**3. [Rule 1 - Bug] Fixed test bodies using positional check_obj instead of check_type kwarg**
- **Found during:** Task 2 GREEN run — test 1 crashed with `BackendNotFoundError: Backend not found for backend, class: (<class '...DataFrameSchema'>, <class 'type'>)`
- **Issue:** The plan's specified test bodies pass `pyspark_sql.DataFrame` (a class) as the `check_obj` positional argument to `get_backend()`. The method calls `type(check_obj)` which gives `type` (the metaclass), not `pyspark_sql.DataFrame`. The registry key is `pyspark_sql.DataFrame`, so no match is found. Polars/ibis tests can pass instances; pyspark requires a SparkSession to create instances.
- **Fix:** Changed `get_backend(pyspark_sql.DataFrame)` to `get_backend(check_type=pyspark_sql.DataFrame)` in all 3 affected tests
- **Files modified:** `tests/narwhals/test_container.py`

**4. [Rule 1 - Bug] Added BACKEND_REGISTRY save/restore for test isolation**
- **Found during:** Task 2 GREEN run — test 2 (`test_pyspark_native_unchanged_when_flag_off`) failed because test 1 already registered the narwhals backend for `pyspark_sql.DataFrame` in the BACKEND_REGISTRY, and `register_backend()` uses first-registration-wins semantics (no overwrite). Clearing the lru_cache alone is insufficient — the registry persists.
- **Fix:** Added `BACKEND_REGISTRY.pop(registry_key, None)` before each test's `register_pyspark_backends()` call, with a `request.addfinalizer` to restore the original value after the test
- **Files modified:** `tests/narwhals/test_container.py`

**5. [Rule 1 - Bug] Added graceful skip for connect test when grpcio-status absent**
- **Found during:** Task 2 test run — `test_pyspark_connect_narwhals_activated_when_opted_in` tried to directly import `from pyspark.sql.connect import dataframe as pyspark_connect` which crashes
- **Fix:** Wrapped the connect import in `try/except Exception: pytest.skip(...)` to handle missing Spark Connect dependencies gracefully
- **Files modified:** `tests/narwhals/test_container.py`

### Pre-existing Test Failures (Out of Scope)

The following 4 tests in `tests/narwhals/test_container.py` were failing BEFORE this plan and remain failing — they are not regressions:

- `test_polars_backends_registered` — pre-existing failure (polars narwhals detection issue)
- `test_polars_narwhals_activated_when_opted_in` — pre-existing failure (same)
- `test_ibis_narwhals_activated_when_opted_in` — pre-existing failure (ibis narwhals detection issue)
- `test_ibis_backend_is_narwhals` — pre-existing failure (same)

These were recorded in `deferred-items.md` for future investigation.

## Pitfalls Encountered

From RESEARCH.md pitfalls:

**Pitfall 1 (builtin_checks import):** Handled — `import pandera.backends.narwhals.builtin_checks  # noqa: F401` included in narwhals branch

**Pitfall 2 (Spark Connect dependencies):** Observed and fixed — `grpcio-status` is not installed in the test environment, causing `pyspark.sql.connect.dataframe` to crash on import even when pyspark >= 3.4 is present. Fixed with try/except in both `register.py` and `types.py`.

**Pitfall 3 (first-registration-wins):** Observed and handled via BACKEND_REGISTRY save/restore pattern in tests.

**Pitfall 4 (test isolation):** Required module-level imports with try/except and request.addfinalizer patterns rather than the simpler forms specified in the plan.

## Verification Commands Run

```bash
# All 4 pyspark tests pass (1 skip)
python -m pytest tests/narwhals/test_container.py -k "pyspark" -v
# Result: 3 passed, 1 skipped

# No new failures in full test_container.py
python -m pytest tests/narwhals/test_container.py -v
# Result: 12 passed, 1 skipped, 4 pre-existing failures

# Success criterion 1: narwhals backend active when flag=True
PANDERA_USE_NARWHALS_BACKEND=True python -c "..."
# Result: SUCCESS

# Success criterion 2: native backend active when flag=False
PANDERA_USE_NARWHALS_BACKEND=False python -c "..."
# Result: SUCCESS

# Syntax validity
python -c "import ast; ast.parse(open('pandera/backends/pyspark/register.py').read())"
# Result: VALID
```

## Threat Flags

None — no new network endpoints, auth paths, file access patterns, or schema changes at trust boundaries introduced. T-01-02 mitigation (ImportError message) implemented as specified.

## Known Stubs

None — the implementation is complete for the plan's scope (REG-01).
