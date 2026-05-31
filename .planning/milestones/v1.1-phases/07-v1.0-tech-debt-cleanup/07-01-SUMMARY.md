---
phase: 07-v1.0-tech-debt-cleanup
plan: "01"
subsystem: testing
tags: [narwhals, ibis, error-handler, polars]

# Dependency graph
requires:
  - phase: 06-eliminate-unnecessary-materialization-lazy-first-failure-cases-and-check-output
    provides: "Phase 6 contract: failure_cases is always native at SchemaError boundary"
provides:
  - "Unified _count_failure_cases using nw.from_native — no backend-specific isinstance branches"
  - "Corrected ibis table type assertion in test_e2e.py (DatabaseTable → Table)"
affects: [error-handler, ibis-integration]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "nw.from_native(obj, eager_only=False).lazy().select(nw.len()).collect() for universal native-type counting"

key-files:
  created: []
  modified:
    - pandera/api/narwhals/error_handler.py
    - tests/backends/narwhals/test_e2e.py

key-decisions:
  - "nw.from_native(failure_cases, eager_only=False) is the correct unified pattern — accepts pl.DataFrame, pl.LazyFrame, and ibis.Table without backend-specific isinstance branches"
  - "Remove _materialize import from error_handler.py — no longer used after nw.from_native unification"

patterns-established:
  - "Use nw.from_native(native, eager_only=False) for universal native-type wrapping at SchemaError boundary"

requirements-completed: []

# Metrics
duration: 5min
completed: 2026-03-24
---

# Phase 7 Plan 01: _count_failure_cases unification and ibis Table assertion fix

**Replaced dead isinstance branch and ibis try/import guard in error_handler.py with a single nw.from_native count; fixed stale ibis DatabaseTable → Table assertion in test_e2e.py**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-24T14:36:04Z
- **Completed:** 2026-03-24T14:41:00Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Eliminated dead `isinstance(failure_cases, (nw.LazyFrame, nw.DataFrame))` branch and ibis-specific try/import/isinstance guard from error_handler.py
- Replaced 25-line implementation with ~12-line unified `nw.from_native(failure_cases, eager_only=False).lazy().select(nw.len()).collect()["len"][0]` count
- Removed unused `_materialize` import from error_handler.py
- Fixed `test_custom_check_receives_table_and_key` assertion from `"DatabaseTable"` to `"Table"` matching ibis's current class name

## Task Commits

Each task was committed atomically:

1. **Task 1: Replace _count_failure_cases dead branch with unified narwhals count** - `931de58` (fix)
2. **Task 2: Fix ibis DatabaseTable → Table assertion in test_e2e.py** - `6ccb4f6` (fix)

## Files Created/Modified

- `pandera/api/narwhals/error_handler.py` - Replaced dead isinstance branches with unified nw.from_native count; removed _materialize import
- `tests/backends/narwhals/test_e2e.py` - Fixed stale assertion: "DatabaseTable" → "Table"

## Decisions Made

- `nw.from_native(failure_cases, eager_only=False)` is the correct unification pattern: `eager_only=False` is mandatory to accept both eager (pl.DataFrame, ibis.Table) and lazy (pl.LazyFrame) native types
- The Phase 6 contract (failure_cases is always native at SchemaError boundary) makes the old isinstance(nw.LazyFrame, nw.DataFrame) branch truly dead — the values were never nw frames at this point

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- `test_custom_check_ibis_lazy` in test_parity.py fails with `TypeError: Unsupported dataframe type, got: <class 'str'>` — confirmed pre-existing failure not caused by this plan's changes (test fails identically on unmodified HEAD).

## Next Phase Readiness

- error_handler.py is clean — no backend-specific logic, no dead branches, no unused imports
- Ready for Plan 07-02 (next tech debt cleanup plan)

---
*Phase: 07-v1.0-tech-debt-cleanup*
*Completed: 2026-03-24*
