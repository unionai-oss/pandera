---
phase: 04-lazy-postprocess-always-lazy-failure-cases
plan: "01"
subsystem: testing
tags: [narwhals, tdd, polars, ibis, failure_cases, lazy]

# Dependency graph
requires:
  - phase: 03-fix-ibischeckbackend-delegation-via-apply-type-dispatch
    provides: NarwhalsCheckBackend apply() dispatch working for polars and ibis builtin checks
provides:
  - RED test stubs for Phase 4 wide-table apply() and lazy postprocess behaviors
  - Updated ibis failure_cases type assertions (nw.DataFrame) establishing RED baseline
  - Updated polars failure_cases type assertions (nw.DataFrame) establishing RED baseline
affects:
  - 04-02-PLAN.md (implementation of wide-table apply() and no-materialization postprocess)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - xfail(strict=False) for stubs that may already pass on some backends
    - TDD RED baseline: assertions drive implementation correctness in plan 04-02

key-files:
  created: []
  modified:
    - tests/backends/narwhals/test_checks.py
    - tests/backends/narwhals/test_e2e.py

key-decisions:
  - "xfail(strict=False) used for polars postprocess stubs because polars path already returns nw.DataFrame (the ibis path is the real bug target)"
  - "TestBuiltinChecksPolars failure_cases assertions updated to nw.DataFrame alongside ibis — both must be RED before Phase 4 removes _to_native"
  - "Pre-existing ibis failures (test_custom_boolean_column_check_passes, test_custom_check_receives_table_and_key) are out-of-scope — logged as pre-existing"

patterns-established:
  - "RED baseline: test_apply_returns_wide_table fails because apply() currently returns narrow 1-column frame"
  - "XFAIL pattern with strict=False: acceptable for stubs where one backend may already satisfy the assertion"

requirements-completed:
  - LAZY-01
  - LAZY-02
  - LAZY-03
  - LAZY-04
  - LAZY-05
  - LAZY-07
  - LAZY-08

# Metrics
duration: 3min
completed: 2026-03-23
---

# Phase 4 Plan 01: Test RED Baseline — Wide Table apply() and Lazy Postprocess Summary

**RED test stubs for Phase 4 behaviors: wide-table apply() columns assertion, nw.DataFrame failure_cases assertions for both polars and ibis paths**

## Performance

- **Duration:** ~3 min
- **Started:** 2026-03-23T01:55:30Z
- **Completed:** 2026-03-23T01:58:30Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Added `test_apply_returns_wide_table` (RED — fails immediately because `apply()` returns narrow 1-column frame, not wide table with data columns)
- Added 4 xfail(strict=False) stubs for postprocess behaviors (polars/ibis no-materialization, ignore_na, n_failure_cases)
- Updated `TestBuiltinChecksIbis` failure_cases assertions to assert `nw.DataFrame` + `ibis.Table` (RED — currently pyarrow.Table)
- Updated `TestBuiltinChecksPolars` failure_cases type assertions to assert `nw.DataFrame` (RED — currently pl.DataFrame after `_to_native`)

## Task Commits

Each task was committed atomically:

1. **Task 1: Add Phase 4 RED test stubs to test_checks.py** - `b51407e` (test)
2. **Task 2: Update failure_cases assertions to nw.DataFrame** - `04eab8e` (test)

**Plan metadata:** (docs commit follows)

## Files Created/Modified
- `tests/backends/narwhals/test_checks.py` - Added 5 new test functions in LAZY-01..08 section
- `tests/backends/narwhals/test_e2e.py` - Updated 4 test methods to assert nw.DataFrame instead of pl.DataFrame/ibis.Table

## Decisions Made

- `xfail(strict=False)` chosen for polars postprocess stubs because the polars path already returns `nw.DataFrame` from `_materialize()` — it wraps polars eagerly. The real bug is on the ibis path (wraps pyarrow.Table). Using strict=False avoids XPASS-as-failure for correct-enough polars behavior.
- Polars `test_isin_fails` updated alongside ibis tests — the plan explicitly asked to scan `TestBuiltinChecksPolars` and update any `isinstance(fc, pl.DataFrame)` assertions.
- Pre-existing failures in `TestCustomChecksIbis` (`test_custom_boolean_column_check_passes`, `test_custom_check_receives_table_and_key`) confirmed as unrelated to this plan — ignored, not fixed.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- `test_postprocess_lazyframe_no_materialization_polars`, `test_ignore_na_lazy`, and `test_n_failure_cases_lazy` showed XPASS on polars because the current polars path already wraps failure_cases as `nw.DataFrame` via `_materialize()`. This is acceptable with `strict=False` — these stubs are primarily targeting the ibis path where the real behavioral change is needed.

## Next Phase Readiness

- RED baseline established: 6 new failing test assertions drive Phase 4 implementation
- Plan 04-02 implements `apply()` wide-table return and `postprocess_lazyframe_output` no-materialization
- All previously-passing tests (171) remain GREEN; new failures are intentional RED baseline

---
*Phase: 04-lazy-postprocess-always-lazy-failure-cases*
*Completed: 2026-03-23*
