---
phase: 03-column-backend
plan: 02
subsystem: backend
tags: [narwhals, polars, ibis, column-validation, backend]

# Dependency graph
requires:
  - phase: 03-01
    provides: test scaffold — 9 xfail tests for ColumnBackend behaviors
  - phase: 02-check-backend
    provides: NarwhalsCheckBackend, _materialize, builtin_checks registration
provides:
  - NarwhalsSchemaBackend (base.py) — subsample, run_check, is_float_dtype helpers
  - ColumnBackend (components.py) — check_nullable, check_unique, check_dtype, run_checks, run_checks_and_handle_errors
  - COLUMN-01 and COLUMN-02 requirements fully met
affects:
  - 04-container-backend (builds on ColumnBackend for DataFrameBackend)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - collect-first pattern for is_duplicated() via _materialize() before calling window functions
    - lazy import of narwhals_engine inside check_dtype to avoid circular imports
    - validate_scope decorator for DATA vs SCHEMA gating on per-column checks
    - _to_native() at every failure_cases construction site to ensure native frames in SchemaErrors

key-files:
  created:
    - pandera/backends/narwhals/base.py
    - pandera/backends/narwhals/components.py
  modified:
    - tests/backends/narwhals/conftest.py

key-decisions:
  - "NarwhalsCheckBackend registered for both nw.LazyFrame (Polars) and nw.DataFrame (Ibis) so run_checks dispatches correctly for both backends"
  - "_materialize stays in checks.py per locked design decision — base.py delegates via NarwhalsCheckBackend._materialize, no duplication"
  - "failure_cases in check_dtype is str(nw_dtype) scalar — consistent with Polars backend convention"

patterns-established:
  - "collect-first: _materialize() before any window function (is_duplicated, is_null) that requires full data visibility"
  - "failure_cases always passed through _to_native() before storing in CoreCheckResult to ensure native frames in SchemaErrors"
  - "lazy import pattern: from pandera.engines import narwhals_engine inside method body to avoid circular import chains"

requirements-completed: [COLUMN-01, COLUMN-02]

# Metrics
duration: 8min
completed: 2026-03-14
---

# Phase 3 Plan 02: Column Backend Summary

**NarwhalsSchemaBackend (base.py) and ColumnBackend (components.py) implementing nullable/unique/dtype/run_checks validation for both Polars LazyFrame and Ibis via narwhals, making all 9 xfail tests from Plan 01 green.**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-14T03:47:54Z
- **Completed:** 2026-03-14T03:55:50Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- NarwhalsSchemaBackend in base.py with subsample(), run_check(), is_float_dtype() — shared helpers for all future backends
- ColumnBackend in components.py with check_nullable (null + NaN detection), check_unique (collect-first pattern), check_dtype (lazy engine import), run_checks, and run_checks_and_handle_errors
- All 18 tests (9 scenarios × 2 backends: polars + ibis) pass; no regressions in Phase 2 check backend tests

## Task Commits

Each task was committed atomically:

1. **Task 1: Create pandera/backends/narwhals/base.py with NarwhalsSchemaBackend** - `139580f` (feat)
2. **Task 2: Create pandera/backends/narwhals/components.py with ColumnBackend** - `7ef35f0` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified
- `pandera/backends/narwhals/base.py` - NarwhalsSchemaBackend with subsample, run_check, is_float_dtype helpers
- `pandera/backends/narwhals/components.py` - ColumnBackend with all per-column validation methods
- `tests/backends/narwhals/conftest.py` - Added nw.DataFrame registration for Ibis backend support

## Decisions Made
- Also register NarwhalsCheckBackend for nw.DataFrame (not just nw.LazyFrame) since Ibis tables wrap as nw.DataFrame — required for run_checks dispatch to work on Ibis backend
- _materialize stays in checks.py, imported via NarwhalsCheckBackend._materialize in base.py — no code duplication

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Register NarwhalsCheckBackend for nw.DataFrame (Ibis)**
- **Found during:** Task 2 (test_run_checks[ibis] failure)
- **Issue:** conftest.py only registered NarwhalsCheckBackend for nw.LazyFrame; Ibis frames are wrapped as nw.DataFrame — dispatch failed with "Backend not found for class: nw.DataFrame"
- **Fix:** Added `Check.register_backend(nw.DataFrame, NarwhalsCheckBackend)` to conftest fixture
- **Files modified:** tests/backends/narwhals/conftest.py
- **Verification:** test_run_checks[ibis] now passes; all 18 tests green
- **Committed in:** 7ef35f0 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 missing critical)
**Impact on plan:** Auto-fix essential for Ibis backend correctness. No scope creep.

## Issues Encountered
- Ibis frames dispatch through nw.DataFrame not nw.LazyFrame — required extending backend registration. This is a structural consequence of how narwhals wraps SQL-lazy (Ibis) tables vs lazy (Polars) frames.

## Next Phase Readiness
- ColumnBackend complete and tested for both Polars and Ibis
- COLUMN-01 and COLUMN-02 requirements fulfilled
- Phase 4 container backend can build on NarwhalsSchemaBackend and ColumnBackend
- No blockers

---
*Phase: 03-column-backend*
*Completed: 2026-03-14*

## Self-Check: PASSED

- pandera/backends/narwhals/base.py: FOUND
- pandera/backends/narwhals/components.py: FOUND
- .planning/phases/03-column-backend/03-02-SUMMARY.md: FOUND
- Commit 139580f (Task 1): FOUND
- Commit 7ef35f0 (Task 2): FOUND
