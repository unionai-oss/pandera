---
phase: 04-lazy-postprocess-always-lazy-failure-cases
plan: "03"
subsystem: narwhals
tags: [narwhals, polars, ibis, lazy, failure_cases, base, run_check, failure_cases_metadata]

# Dependency graph
requires:
  - phase: 04-lazy-postprocess-always-lazy-failure-cases
    plan: "02"
    provides: wide-table apply(), fully lazy postprocess_lazyframe_output, nw.DataFrame failure_cases from checks
provides:
  - run_check narwhals path keeps failure_cases as nw.DataFrame (not native pl.DataFrame)
  - For polars-backed frames: nw.LazyFrame collected to nw.DataFrame via .collect()
  - For ibis-backed frames: nw.DataFrame wrapping ibis.Table kept lazy (not materialized)
  - failure_cases_metadata uses single universal nw.DataFrame branch via _materialize + to_arrow() + pl.from_arrow()
  - Zero backend-specific isinstance checks (ibis.Table, pyarrow.Table, pl.DataFrame) in failure_cases_metadata frame-handling path
  - NarwhalsErrorHandler._count_failure_cases handles nw.DataFrame wrapping ibis.Table
  - TestBuiltinChecksIbis failure_cases_type and failure_cases_values GREEN
  - TestBuiltinChecksPolars all GREEN (zero regressions)
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "nw.DataFrame wrapping ibis.Table: kept lazy in run_check (no _materialize), materialized in failure_cases_metadata via _materialize(err.failure_cases)"
    - "to_arrow() + pl.from_arrow(): backend-agnostic conversion from narwhals to polars — works for polars-backed and ibis-backed frames"
    - "LazyFrame vs DataFrame distinction: polars failure_cases is nw.LazyFrame (collect to nw.DataFrame); ibis failure_cases is nw.DataFrame (keep as-is)"

key-files:
  created: []
  modified:
    - pandera/backends/narwhals/base.py
    - pandera/api/narwhals/error_handler.py
    - tests/backends/narwhals/test_container.py
    - tests/backends/narwhals/test_parity.py

key-decisions:
  - "polars vs ibis distinction in run_check: isinstance(fc, nw.LazyFrame) distinguishes polars (LazyFrame from filter on LazyFrame) from ibis (DataFrame wrapping ibis.Table from filter on interchange DataFrame) — polars gets .collect(), ibis kept as-is"
  - "to_arrow() + pl.from_arrow() in failure_cases_metadata: backend-agnostic materialization; _materialize(err.failure_cases) produces an eager narwhals frame, to_arrow() extracts Arrow table, pl.from_arrow() converts to polars — no ibis/pyarrow isinstance checks needed"
  - "NarwhalsErrorHandler._count_failure_cases extended: nw.DataFrame wrapping ibis.Table cannot be len()-counted; unwrap to native and call .count().to_pyarrow().as_py() via ibis API"
  - "test_failure_cases_is_native and test_failure_cases_native_ibis updated: Phase 4 changes the contract — failure_cases is now nw.DataFrame not native pl.DataFrame/pyarrow.Table"

patterns-established:
  - "Narwhals failure_cases pattern: keep as nw.DataFrame through run_check; failure_cases_metadata materializes uniformly — never unwrap to native before SchemaError construction"
  - "Backend-agnostic conversion: _materialize(nw_frame).to_arrow() → pl.from_arrow() works for any narwhals backend (polars, ibis, pandas)"

requirements-completed:
  - LAZY-04
  - LAZY-05
  - LAZY-06

# Metrics
duration: 7min
completed: 2026-03-23
---

# Phase 4 Plan 03: Narwhals-agnostic run_check + failure_cases_metadata Summary

**Backend-agnostic run_check narwhals path (failure_cases as nw.DataFrame) and single-branch failure_cases_metadata via _materialize + to_arrow() + pl.from_arrow() — zero ibis/pyarrow isinstance checks**

## Performance

- **Duration:** ~7 min
- **Started:** 2026-03-23T02:22:09Z
- **Completed:** 2026-03-23T02:29:00Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- `run_check` narwhals path no longer calls `_to_native` on `failure_cases` — it's kept as `nw.DataFrame`
- For polars: `nw.LazyFrame` from `filter()` is collected to `nw.DataFrame` via `.collect()`
- For ibis: `nw.DataFrame` wrapping `ibis.Table` is kept lazy (not materialized via `.execute()`)
- `failure_cases_metadata` rewritten with single universal branch: `_materialize` + `to_arrow()` + `pl.from_arrow()`
- Zero `ibis.Table`, `pyarrow.Table`, or `pl.DataFrame` isinstance checks in the frame-handling path
- `NarwhalsErrorHandler._count_failure_cases` extended to handle `nw.DataFrame` wrapping `ibis.Table`
- TestBuiltinChecksPolars: all 7 tests GREEN; TestBuiltinChecksIbis: all 4 tests GREEN including failure_cases_type + values

## Task Commits

Each task was committed atomically:

1. **Task 1: Remove _to_native from run_check narwhals path failure_cases** - `5b69446` (feat)
   - Refinement commit: `e3dcb10` (fix)
2. **Task 2: Narwhals-ify failure_cases_metadata — collapse all branches into one** - `8791709` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified
- `pandera/backends/narwhals/base.py` - (1) run_check: remove _to_native on failure_cases, add LazyFrame → DataFrame collection for polars; (2) failure_cases_metadata: replace three branches (pl.DataFrame, ibis, pyarrow) with single universal nw.DataFrame path via _materialize + to_arrow() + pl.from_arrow(); updated docstring
- `pandera/api/narwhals/error_handler.py` - _count_failure_cases extended to handle nw.DataFrame wrapping ibis.Table
- `tests/backends/narwhals/test_container.py` - test_failure_cases_is_native: updated from pl.DataFrame assertion to nw.DataFrame assertion (Phase 4 behavior change)
- `tests/backends/narwhals/test_parity.py` - test_failure_cases_native_ibis: updated from "must be native" to "must be nw.DataFrame wrapping ibis.Table" (Phase 4 behavior change)

## Decisions Made

- **polars vs ibis distinction in run_check:** `isinstance(fc, nw.LazyFrame)` cleanly distinguishes polars failure_cases (result of `.filter()` on a polars `nw.LazyFrame` — produces `nw.LazyFrame`) from ibis failure_cases (result of `.filter()` on an ibis-backed `nw.DataFrame` — stays `nw.DataFrame`). Polars gets `.collect()`, ibis kept as-is.
- **to_arrow() + pl.from_arrow() pattern:** The plan specified this backend-agnostic conversion. `_materialize(err.failure_cases)` produces an eager narwhals frame (via `.collect()` for polars, via `native.execute()` for ibis), then `to_arrow()` extracts an Arrow table, then `pl.from_arrow()` converts to polars for `pl.concat`. Works identically for both backends.
- **NarwhalsErrorHandler fix:** `len(nw.DataFrame wrapping ibis.Table)` raises `AttributeError: 'IbisLazyFrame' object has no attribute '__len__'`. Added detection path in `_count_failure_cases` to unwrap via `nw.to_native()` and use `ibis.Table.count().to_pyarrow().as_py()`.
- **Test updates:** `test_failure_cases_is_native` and `test_failure_cases_native_ibis` were written for the pre-Phase-4 contract (native types). These tests are in-scope (directly caused by Task 1 changes) and were updated to reflect the new Phase 4 contract.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] nw.LazyFrame vs nw.DataFrame distinction needed in run_check**
- **Found during:** Task 1 (Remove _to_native from run_check narwhals path)
- **Issue:** The plan's replacement code says `failure_cases = fc` but `check_result.failure_cases` for polars is `nw.LazyFrame` (from `.filter()` on a LazyFrame), not `nw.DataFrame`. Test `test_greater_than_fails_failure_cases_type` requires `isinstance(fc, nw.DataFrame)`.
- **Fix:** Added `if isinstance(fc, nw.LazyFrame): fc = fc.collect()` to convert polars to eager DataFrame while leaving ibis-backed nw.DataFrame (already eager interchange) unchanged.
- **Files modified:** `pandera/backends/narwhals/base.py`
- **Verification:** TestBuiltinChecksPolars 7/7 GREEN; TestBuiltinChecksIbis failure_cases_type asserts native is ibis.Table (not pandas).
- **Committed in:** `e3dcb10` (Task 1 refinement commit)

**2. [Rule 1 - Bug] NarwhalsErrorHandler._count_failure_cases fails for nw.DataFrame wrapping ibis.Table**
- **Found during:** Task 2 (Narwhals-ify failure_cases_metadata)
- **Issue:** `len(nw.DataFrame wrapping ibis.Table)` raises `AttributeError: 'IbisLazyFrame' has no attribute '__len__'`. This is triggered during ibis lazy validation error collection.
- **Fix:** Extended `_count_failure_cases` to detect `nw.DataFrame` wrapping `ibis.Table` via `nw.to_native()` and use `ibis.Table.count().to_pyarrow().as_py()`.
- **Files modified:** `pandera/api/narwhals/error_handler.py`
- **Verification:** `TestLazyValidationIbis::test_ibis_lazy_collects_multiple_errors` and `test_ibis_lazy_failure_cases_is_dataframe` GREEN.
- **Committed in:** `8791709` (Task 2 commit)

**3. [Rule 1 - Bug] test_failure_cases_is_native and test_failure_cases_native_ibis: stale pre-Phase-4 assertions**
- **Found during:** Task 2 (overall test verification)
- **Issue:** Both tests assert the OLD pre-Phase-4 contract (failure_cases is native `pl.DataFrame` or non-narwhals). Phase 4 changes the contract: failure_cases is now `nw.DataFrame`.
- **Fix:** Updated both tests to assert `isinstance(fc, nw.DataFrame)` and for ibis `isinstance(nw.to_native(fc), ibis.Table)`.
- **Files modified:** `tests/backends/narwhals/test_container.py`, `tests/backends/narwhals/test_parity.py`
- **Verification:** Both tests GREEN with new assertions.
- **Committed in:** `8791709` (Task 2 commit)

---

**Total deviations:** 3 auto-fixed (3 Rule 1 bugs)
**Impact on plan:** All fixes directly caused by Task 1 changes and necessary for correctness. No scope creep.

## Issues Encountered

- Pre-existing test failure: `TestCustomChecksIbis::test_custom_check_receives_table_and_key` asserts `table_type == "DatabaseTable"` but ibis changed the class name to `"Table"` in a newer version. Logged to `deferred-items.md`. Not caused by Phase 04-03 changes.

## Next Phase Readiness

- Phase 04 complete: failure_cases is now `nw.DataFrame` end-to-end for both polars and ibis
- `failure_cases_metadata` is fully backend-agnostic with zero ibis/pyarrow isinstance checks
- The narwhals backend now provides a clean abstraction layer: checks produce nw.DataFrame, SchemaError carries nw.DataFrame, failure_cases_metadata materializes uniformly
- No blocking issues for future phases

---
*Phase: 04-lazy-postprocess-always-lazy-failure-cases*
*Completed: 2026-03-23*
