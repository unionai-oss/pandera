---
phase: 04-lazy-postprocess-always-lazy-failure-cases
plan: "02"
subsystem: checks
tags: [narwhals, polars, ibis, lazy, failure_cases, apply, postprocess, wide-table]

# Dependency graph
requires:
  - phase: 04-lazy-postprocess-always-lazy-failure-cases
    plan: "01"
    provides: RED test stubs for wide-table apply() and lazy postprocess behaviors
provides:
  - apply() returns wide table (frame + CHECK_OUTPUT_KEY) for all non-bool return paths
  - postprocess_lazyframe_output is fully lazy — no _materialize calls
  - Polars builtin checks regression: all 28 pass/fail tests GREEN
  - Ibis builtin checks regression: all 28 pass/fail tests GREEN
  - test_apply_returns_wide_table flips from RED to GREEN (both polars and ibis)
  - test_postprocess_lazyframe_no_materialization_ibis flips from XFAIL to XPASS
affects:
  - 04-03-PLAN.md (base.py changes to consume wide-table check_output)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "ibis wide-table: row_number join via native ibis API to attach CHECK_OUTPUT_KEY to original ibis Table"
    - "Backend detection: hasattr(nw.to_native(out), 'execute') distinguishes ibis from polars"
    - "Polars path: collect() 1-col bool result (tiny) and attach as Series; LazyFrame stays lazy"

key-files:
  created: []
  modified:
    - pandera/backends/narwhals/checks.py
    - tests/backends/narwhals/test_checks.py

key-decisions:
  - "ibis row_number join: narwhals cannot pass a Series from one ibis relation into with_columns of another; native ibis row_number().over(window()) join is the correct approach to attach CHECK_OUTPUT_KEY to the original ibis Table while keeping it lazy"
  - "Backend detection via hasattr(native, 'execute'): polars pl.LazyFrame has no .execute(); ibis.Table does — this cleanly separates the two paths without importing polars or ibis directly"
  - "element_wise .select(selector) kept: plan edit suggested dropping it, but removing would send all original data columns through all_horizontal which only works for all-boolean frames; narrow extraction before wide-table re-attachment is necessary"
  - "test_builtin_checks_pass/fail updated: ibis-backed nw.DataFrame requires execute() via native rather than narwhals indexing ([0] fails for ibis interchange series)"

patterns-established:
  - "Wide table pattern: apply() returns check_obj.frame + CHECK_OUTPUT_KEY; postprocess_lazyframe_output uses check_output.filter/select to stay lazy"
  - "Ibis join-based column attachment: when narwhals with_columns rejects cross-relation series, use row_number join as the canonical ibis-safe pattern"

requirements-completed:
  - LAZY-01
  - LAZY-02
  - LAZY-03
  - LAZY-07
  - LAZY-08

# Metrics
duration: 17min
completed: 2026-03-23
---

# Phase 4 Plan 02: Wide-Table apply() and Lazy postprocess_lazyframe_output Summary

**apply() now returns the full wide table (frame + CHECK_OUTPUT_KEY) for both polars and ibis; postprocess_lazyframe_output is fully lazy with no _materialize calls — ibis failure_cases stays as ibis.Table**

## Performance

- **Duration:** ~17 min
- **Started:** 2026-03-23T02:00:57Z
- **Completed:** 2026-03-23T02:18:00Z
- **Tasks:** 1
- **Files modified:** 2

## Accomplishments
- `apply()` rewired to return wide table (original frame + CHECK_OUTPUT_KEY column) for all non-bool return paths — enables postprocess to filter lazily without materializing
- `postprocess_lazyframe_output` rewritten with zero `_materialize` calls: uses `check_output.filter(~nw.col(CHECK_OUTPUT_KEY))` and `check_output.select(nw.col(CHECK_OUTPUT_KEY).all())` staying fully lazy
- `test_apply_returns_wide_table` flips GREEN for both polars and ibis (was RED baseline from Plan 04-01)
- `test_postprocess_lazyframe_no_materialization_ibis` XPASS — ibis failure_cases is now nw.DataFrame wrapping ibis.Table (lazy, not pyarrow.Table)
- All 28 Polars + 28 Ibis builtin check pass/fail tests remain GREEN (zero regressions)

## Task Commits

Each task was committed atomically:

1. **Task 1: Rewrite apply() to return wide table + lazy postprocess_lazyframe_output** - `4ee489b` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified
- `pandera/backends/narwhals/checks.py` - Three edits: (1) apply() final return using wide-table via polars collect+attach or ibis row_number join; (2) postprocess_lazyframe_output fully lazy (no _materialize calls); (3) element_wise .select(selector) kept (see deviations)
- `tests/backends/narwhals/test_checks.py` - test_builtin_checks_pass/fail updated to handle ibis-backed nw.DataFrame via native .execute() instead of narwhals interchange indexing

## Decisions Made

- **ibis wide-table via row_number join**: narwhals cannot pass a Series from one ibis relation into `with_columns` of another (fails with `IbisExpr._from_series` not found). The native ibis API `row_number().over(window())` join is the canonical approach. Both the primary and fallback paths in `apply()` use this join pattern.
- **Backend detection without importing backends**: `hasattr(nw.to_native(out), "execute")` reliably identifies ibis-backed frames (both nw.LazyFrame and nw.DataFrame wrapping ibis) vs polars (pl.LazyFrame has no `.execute()`). Avoids conditional imports.
- **element_wise .select(selector) kept**: Plan Edit 1 said to drop `.select(selector)` from the element_wise branch. Removing it would produce a multi-column frame where non-bool data columns would be fed through `all_horizontal`, which only works for all-boolean frames and would produce nonsensical results for integer data. The narrow selection is necessary before wide-table re-attachment — this is a plan specification error, not scope creep.
- **test_builtin_checks_pass/fail ibis handling**: After postprocess changes, ibis `check_passed` is an ibis-backed nw.DataFrame whose interchange series doesn't support `[0]` indexing. Added `nw.to_native(passed).execute()[col].iloc[0]` branch for the ibis case (Rule 1 auto-fix — prevented regression).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] test_builtin_checks_pass/fail regression: ibis interchange series indexing**
- **Found during:** Task 1 (postprocess_lazyframe_output rewrite)
- **Issue:** After postprocess_lazyframe_output stopped materializing, `check_passed` for ibis became an ibis-backed nw.DataFrame. The test code `passed[CHECK_OUTPUT_KEY][0]` uses narwhals indexing which fails with `NotImplementedError: item is not supported for interchange-level dataframes`.
- **Fix:** Added ibis-path handling in test_builtin_checks_pass/fail: `native.execute()[CHECK_OUTPUT_KEY].iloc[0]` for frames where `nw.to_native(passed)` has `.execute()`.
- **Files modified:** `tests/backends/narwhals/test_checks.py`
- **Verification:** All 28 polars + 28 ibis builtin check tests GREEN.
- **Committed in:** `4ee489b` (Task 1 commit)

**2. [Rule 1 - Bug] element_wise branch: plan Edit 1 specification error**
- **Found during:** Task 1 (apply() element_wise branch analysis)
- **Issue:** Plan said to drop `.select(selector)` after `with_columns(map_batches)`. Removing it would send all original data columns (including non-boolean integers) through `all_horizontal`, which requires all-boolean inputs and would produce wrong results.
- **Fix:** Kept `.select(selector)` in element_wise branch. The wide-table return is still achieved via the final `check_obj.frame.with_columns(out.collect()[CHECK_OUTPUT_KEY])`.
- **Files modified:** `pandera/backends/narwhals/checks.py`
- **Verification:** All builtin check tests pass; no change in element_wise behavior from user perspective.
- **Committed in:** `4ee489b` (Task 1 commit)

**3. [Rule 1 - Bug] ibis e2e regression: components.py converts ibis frame to nw.LazyFrame**
- **Found during:** Task 1 (e2e test run after implementation)
- **Issue:** `components.py` calls `check_lf = check_lf.lazy()` for ibis frames, converting nw.DataFrame to nw.LazyFrame. The initial implementation detected ibis via `isinstance(out, nw.DataFrame)` which missed the ibis-backed LazyFrame case — caused `_from_series` error in the polars fallback path.
- **Fix:** Changed backend detection to `hasattr(nw.to_native(out), "execute")` which correctly identifies ibis-backed frames regardless of whether they are nw.LazyFrame or nw.DataFrame.
- **Files modified:** `pandera/backends/narwhals/checks.py`
- **Verification:** `test_greater_than_passes[ibis]` and `test_isin_passes[ibis]` pass; no new e2e regressions.
- **Committed in:** `4ee489b` (Task 1 commit)

---

**Total deviations:** 3 auto-fixed (2 Rule 1 bugs, 1 Rule 1 plan specification error)
**Impact on plan:** All fixes necessary for correctness. The element_wise deviation is a plan spec error — the plan's intended behavior (wide-table apply) is fully achieved through the kept `.select(selector)`. No scope creep.

## Issues Encountered

- The ibis narwhals backend does not support `_from_series` for cross-relation column attachment — the row_number join workaround is non-trivial but necessary for lazy ibis behavior.
- `components.py` calling `.lazy()` on ibis frames creates an ibis-backed `nw.LazyFrame`, requiring backend detection at the `nw.to_native()` level rather than via isinstance checks.

## Next Phase Readiness

- Wide table from `apply()` is established; `postprocess_lazyframe_output` is lazy end-to-end.
- Plan 04-03 (`base.py`) needs to consume `check_result.check_output` as a narwhals frame (not ibis-native), update `run_check` to handle the new ibis-backed nw.DataFrame/LazyFrame check results, and update `failure_cases` propagation so e2e tests `test_greater_than_fails_failure_cases_type/values` flip GREEN.
- Pre-existing ibis failures in test_e2e.py (failure_cases type/value assertions) remain RED — these are plan 04-03 targets.

---
*Phase: 04-lazy-postprocess-always-lazy-failure-cases*
*Completed: 2026-03-23*
