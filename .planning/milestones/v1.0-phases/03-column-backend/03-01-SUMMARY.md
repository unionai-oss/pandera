---
phase: 03-column-backend
plan: 01
subsystem: testing
tags: [narwhals, pytest, xfail, column-backend, tdd]

requires:
  - phase: 02-check-backend
    provides: "NarwhalsCheckBackend, conftest fixtures (make_narwhals_frame, narwhals check backend registration)"

provides:
  - "test_components.py with 9 xfail stubs covering check_nullable, check_unique, check_dtype, run_checks"
  - "Parameterized test scaffold against polars + ibis via make_narwhals_frame"
  - "Nyquist compliance gate for Plan 03-02 ColumnBackend implementation"

affects:
  - 03-column-backend

tech-stack:
  added: []
  patterns:
    - "xfail(ColumnBackend is None, strict=False) guards tests before implementation exists — same pattern as conftest.py"
    - "SimpleNamespace schema stub for testing — lightweight alternative to full pandera Column instantiation"
    - "Module-level try/except import guard allows file collection before components.py lands"

key-files:
  created:
    - tests/backends/narwhals/test_components.py
  modified: []

key-decisions:
  - "9 test functions x 2 backends (polars + ibis) = 18 collected items — fixture parameterization doubles coverage automatically"
  - "Used SimpleNamespace schema stub instead of real pandera Column to avoid coupling test scaffold to Column API before it is stable"
  - "Float NaN test uses pl.Series([...]) directly rather than a dict literal to force Float64 dtype in Polars"

patterns-established:
  - "TDD RED phase: write tests before implementation, verify all xfail, commit, then implement in next plan"
  - "Import guard pattern: try/except at module level allows collection even when implementation missing"

requirements-completed:
  - COLUMN-01
  - COLUMN-02

duration: 5min
completed: 2026-03-13
---

# Phase 3 Plan 01: Column Backend Test Scaffold Summary

**9 xfail test stubs covering check_nullable (3), check_unique (2), check_dtype (3), and run_checks (1), parameterized across polars and ibis via make_narwhals_frame — 18 items collected, 18 xfailed, 0 errors**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-14T03:44:59Z
- **Completed:** 2026-03-14T03:49:00Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Created `tests/backends/narwhals/test_components.py` with 9 test functions covering all ColumnBackend behaviors
- All 18 test cases (9 functions x 2 backends) collect without import errors and run as xfail
- Established the Nyquist compliance gate — Plan 03-02 can now verify its implementation by running the same file

## Task Commits

Each task was committed atomically:

1. **Task 1: Write test_components.py with xfail stubs for all column backend behaviors** - `eb1df9c` (test)

**Plan metadata:** _(to be committed)_

## Files Created/Modified
- `tests/backends/narwhals/test_components.py` - 9 xfail test stubs for ColumnBackend (check_nullable, check_unique, check_dtype, run_checks), parameterized for polars + ibis

## Decisions Made
- Used `SimpleNamespace` schema stub instead of a real pandera `Column` to decouple the test scaffold from Column API details (which are not yet finalized)
- Float NaN test creates the frame using `pl.Series([1.0, float("nan"), 3.0])` to ensure Float64 dtype is inferred correctly
- `strict=False` on xfail allows the tests to pass as xfail even if execution unexpectedly succeeds (matches the pattern established in Phase 2)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Test scaffold is in place; Plan 03-02 can implement `ColumnBackend` and verify against this file
- All tests should flip from xfail to pass once `pandera/backends/narwhals/components.py` is created

---
*Phase: 03-column-backend*
*Completed: 2026-03-13*
