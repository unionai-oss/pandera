---
phase: 09-accumulate-check-outputs-into-single-wide-table-for-narwhals-idiomatic-drop-invalid-rows
plan: "01"
subsystem: testing
tags: [narwhals, polars, ibis, drop_invalid_rows, tdd, red-baseline]

# Dependency graph
requires: []
provides:
  - RED baseline documenting 20 failing drop_invalid_rows tests (8 polars lazy=True, 12 ibis)
  - xfail(strict=True) parity test stub asserting check_output will be nw.Expr after Phase 09-02
  - Narwhals backend suite confirmed all-green (208 passed, 2 xfailed, 8 skipped)
affects:
  - 09-02-PLAN.md (GREEN implementation plan — this RED baseline is its before-state)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "xfail(strict=True) to document future-behavior contract while keeping suite green"
    - "TDD RED baseline: confirm failure mode before any production code changes"

key-files:
  created: []
  modified:
    - tests/backends/narwhals/test_parity.py

key-decisions:
  - "xfail(strict=True) chosen over skip or xfail(strict=False) — enforces that the test actually fails now (CI breaks if it starts passing unexpectedly before 09-02)"
  - "Polars failure mode: TypeError: Slicing is not supported on LazyFrame — drop_invalid_rows slices nw.LazyFrame wide table at column index"
  - "Ibis failure mode: AttributeError: LazyFrame has no attribute mutate — IbisSchemaBackend delegation receives nw.LazyFrame instead of ibis.Table"

patterns-established:
  - "Phase 09 RED/GREEN TDD pattern: stub xfail test asserting post-fix contract, then remove xfail in GREEN plan"

requirements-completed: [DIR-01, DIR-02, DIR-03, DIR-04]

# Metrics
duration: 2min
completed: 2026-03-24
---

# Phase 09 Plan 01: RED Baseline Summary

**xfail parity stub committed confirming 20 drop_invalid_rows failures (8 polars lazy=True TypeError, 12 ibis AttributeError) with narwhals suite all-green at 208 passed**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-25T05:21:28Z
- **Completed:** 2026-03-25T05:23:31Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Confirmed 20 failing drop_invalid_rows tests: 8 polars (lazy=True only) and 12 ibis (duckdb + sqlite)
- Documented exact failure modes: polars crashes with `TypeError: Slicing is not supported on LazyFrame`; ibis crashes with `AttributeError: 'LazyFrame' object has no attribute 'mutate'`
- Confirmed narwhals backend suite all-green: 208 passed, 2 xfailed, 8 skipped (zero regressions)
- Added `test_drop_invalid_rows_expr_accumulation` xfail(strict=True) parity stub to test_parity.py documenting expected post-fix behavior

## Task Commits

Each task was committed atomically:

1. **Task 1: Run RED baseline and add parity test stub** - `2d1c055` (test)

**Plan metadata:** (docs commit — see state_updates below)

_Note: TDD RED tasks produce a single test commit establishing baseline_

## Files Created/Modified
- `tests/backends/narwhals/test_parity.py` - Added `test_drop_invalid_rows_expr_accumulation` xfail(strict=True) test asserting check_output is nw.Expr (documents Phase 09-02 post-fix contract)

## Decisions Made
- `xfail(strict=True)` chosen over `xfail(strict=False)`: strict=True ensures CI breaks if the test starts unexpectedly passing before the 09-02 fix lands — prevents silent accumulation of stale marks
- Assertion targets `nw.Expr` type: this is the post-fix contract from Phase 09-02 where apply() returns expressions directly rather than wide LazyFrame tables

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- RED baseline committed; 09-02 has a documented before-state to validate against
- The xfail test will be promoted to a strict passing test in 09-02 when apply() returns nw.Expr
- Production code to change: pandera/backends/narwhals/checks.py (apply/postprocess) and pandera/backends/narwhals/base.py (drop_invalid_rows)

---
*Phase: 09-accumulate-check-outputs-into-single-wide-table-for-narwhals-idiomatic-drop-invalid-rows*
*Completed: 2026-03-24*
