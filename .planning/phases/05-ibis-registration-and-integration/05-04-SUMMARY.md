---
phase: 05-ibis-registration-and-integration
plan: "04"
subsystem: testing
tags: [narwhals, ibis, checks, backend, delegation, failure_cases]

requires:
  - phase: 05-01
    provides: register_ibis_backends() wiring ibis.Table to narwhals DataFrameSchemaBackend
  - phase: 05-02
    provides: NarwhalsCheckBackend.apply() with Dispatcher-based builtin routing
  - phase: 05-03
    provides: ibis narwhals fixture + xfail stubs converted to passing tests

provides:
  - NarwhalsCheckBackend.__call__() ibis delegation path for user-defined checks
  - run_check in base.py that preserves ibis.Table failure_cases without materializing
affects:
  - tests/ibis/ test suite (closes Gap 1 and Gap 3 from 05-VERIFICATION.md)
  - future ibis check contract consumers expecting IbisData wrapping

tech-stack:
  added: []
  patterns:
    - "Lazy ibis import (try/except ImportError) inside hot paths to keep ibis optional"
    - "Dispatcher._function_registry[NarwhalsData] lookup to distinguish builtin vs user-defined checks"
    - "element_wise guard before ibis delegation — preserve narwhals NotImplementedError contract"

key-files:
  created: []
  modified:
    - pandera/backends/narwhals/checks.py
    - pandera/backends/narwhals/base.py

key-decisions:
  - "element_wise=True checks skip ibis delegation so apply() raises NotImplementedError for SQL-lazy backends (ibis would attempt UDF instead)"
  - "ibis BooleanScalar check_passed evaluated via .execute() in run_check — avoids _materialize() which converts to narwhals then polars"
  - "failure_cases returned as ibis.Table directly from run_check ibis path — tests/ibis/ call .execute()/.to_pandas() on them"
  - "check_output also returned as-is (ibis.Table) in ibis path of run_check — lazy ibis contract preserved end-to-end"

patterns-established:
  - "Ibis path detection: isinstance(check_result.check_passed, (_ir.BooleanScalar, _ir.BooleanColumn)) or isinstance(failure_cases, ibis.Table)"
  - "Two-branch run_check: _is_ibis_result bool gates ibis path vs narwhals materialize path"

requirements-completed: [REGISTER-03, TEST-02]

duration: 4min
completed: 2026-03-15
---

# Phase 5 Plan 4: Ibis Check Delegation and Failure Cases Gap Closure Summary

**NarwhalsCheckBackend delegates user-defined ibis checks to IbisCheckBackend (preserving IbisData wrapping), and run_check skips failure_cases materialization for ibis results (returning lazy ibis.Table instead of pyarrow)**

## Performance

- **Duration:** ~4 min
- **Started:** 2026-03-15T05:38:29Z
- **Completed:** 2026-03-15T05:42:00Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Closed Gap 1 from VERIFICATION.md: NarwhalsCheckBackend now detects ibis input and delegates user-defined checks to IbisCheckBackend, which wraps the table in IbisData(table, key) as ibis check functions expect
- Closed Gap 3 from VERIFICATION.md: run_check in base.py detects ibis BooleanScalar check_passed and skips _materialize(), preserving ibis.Table failure_cases for callers that call .execute()/.to_pandas()
- element_wise guard added to ibis delegation — ensures narwhals path raises NotImplementedError for SQL-lazy backends (ibis would have tried to create a UDF instead)

## Task Commits

Each task was committed atomically:

1. **Task 1: Add ibis delegation path to NarwhalsCheckBackend.__call__()** - `0eb3ea5` (feat)
2. **Task 2: Make run_check in base.py ibis-aware (skip failure_cases materialization)** - `1c566b8` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified
- `pandera/backends/narwhals/checks.py` - Added ibis delegation block in __call__(); element_wise guard; lazy IbisCheckBackend import
- `pandera/backends/narwhals/base.py` - Added _is_ibis_result detection; ibis branch in run_check that calls .execute() and preserves ibis.Table failure_cases

## Decisions Made
- `element_wise=True` checks skip ibis delegation so the narwhals `apply()` path raises `NotImplementedError` for SQL-lazy backends; IbisCheckBackend would silently attempt to wrap the lambda as a UDF and fail with `MissingReturnAnnotationError`
- ibis `check_passed` is evaluated via `.execute()` directly in `run_check` rather than going through `_materialize()` — `_materialize()` wraps via narwhals then collects, losing the ibis BooleanScalar type
- `failure_cases` and `check_output` returned as-is (lazy ibis.Table) from the ibis branch — downstream ibis tests call `.execute()/.to_pandas()` and require the lazy type

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Added element_wise guard before ibis delegation**
- **Found during:** Task 1 verification (narwhals test suite)
- **Issue:** `test_element_wise_sql_lazy_raises[ibis]` failed because element_wise lambda was delegated to IbisCheckBackend, which tried to create a UDF from it and raised `MissingReturnAnnotationError` instead of `NotImplementedError`
- **Fix:** Added `and not self.check.element_wise` to the delegation condition, so element_wise checks fall through to the narwhals apply() path which raises the correct error
- **Files modified:** pandera/backends/narwhals/checks.py
- **Verification:** narwhals test suite passes 124 tests including `test_element_wise_sql_lazy_raises[ibis]`
- **Committed in:** `0eb3ea5` (part of Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Fix required for correctness — prevents wrong exception type being raised for element_wise ibis checks. No scope creep.

## Issues Encountered
- 5 pre-existing failures in `tests/ibis/test_ibis_check.py` remain unchanged: 3 `test_element_wise_*` tests (ibis UDF limitation) and 2 `test_*_n_failure_cases` tests (Gap 2: `len()` called on ibis.Table in error_handler.py `_count_failure_cases`). These are out of scope for this plan (Gap 2 is not addressed here).

## Next Phase Readiness
- Gap 1 and Gap 3 from VERIFICATION.md are closed
- Gap 2 (failure_cases_metadata crash on ibis Object dtype in pl.concat) remains — addressed separately
- The ibis check behavioral contract (IbisData wrapping, lazy failure_cases) is restored for user-defined checks going through the narwhals backend

---
*Phase: 05-ibis-registration-and-integration*
*Completed: 2026-03-15*
