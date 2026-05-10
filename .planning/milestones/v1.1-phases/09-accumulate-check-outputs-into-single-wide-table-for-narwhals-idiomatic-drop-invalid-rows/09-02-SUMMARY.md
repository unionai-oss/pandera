---
phase: 09-accumulate-check-outputs-into-single-wide-table-for-narwhals-idiomatic-drop-invalid-rows
plan: 02
subsystem: validation
tags: [narwhals, polars, ibis, drop_invalid_rows, nw.Expr, nw.all_horizontal, lazy-validation]

# Dependency graph
requires:
  - phase: 09-01
    provides: RED baseline with 20 failing drop_invalid_rows tests and xfail parity test stub

provides:
  - apply() in checks.py returns nw.Expr directly for native=False and element_wise paths
  - postprocess_expr_output() stores expr+deferred failure_cases (no wide table during check loop)
  - drop_invalid_rows() in base.py uses nw.all_horizontal accumulation — pure narwhals, no backend delegation
  - failure_cases_metadata() reconstructs failure_cases from stored nw.Expr when SchemaErrors is raised
  - SCHEMA_AND_DATA depth forced in container validate() when drop_invalid_rows=True (LazyFrame fix)
  - parity test promoted from xfail(strict=True) to strict passing

affects:
  - Any future phase modifying checks.py apply() or postprocess() dispatch
  - Any phase adding new check_output types beyond nw.Expr / nw.LazyFrame / bool

# Tech tracking
tech-stack:
  added: []
  patterns:
    - nw.all_horizontal accumulation for multi-check row filtering — single wide frame per validation
    - Deferred failure_cases: postprocess_expr_output stores None, failure_cases_metadata reconstructs once
    - config_context(SCHEMA_AND_DATA) in container.validate() when drop_invalid_rows=True

key-files:
  created: []
  modified:
    - pandera/backends/narwhals/checks.py
    - pandera/backends/narwhals/base.py
    - pandera/backends/narwhals/container.py
    - tests/backends/narwhals/test_parity.py

key-decisions:
  - "apply() returns nw.Expr directly (not wide table) for native=False and element_wise paths"
  - "postprocess_expr_output() stores check_output=expr, failure_cases=None — no wide table in check loop"
  - "drop_invalid_rows() uses nw.all_horizontal(*exprs) for row filter — no ibis delegation, no polars isinstance"
  - "SERIES_CONTAINS_NULLS errors handled in drop_invalid_rows by reconstructing ~nw.col(selector).is_null()"
  - "container.validate() uses config_context(SCHEMA_AND_DATA) when drop_invalid_rows=True — LazyFrame defaults to SCHEMA_ONLY"
  - "ignore_na applied at column level AFTER evaluation (not as expr|expr.is_null()) to avoid ibis SQL nullability bugs"
  - "failure_cases_metadata() reconstructs failure_cases from nw.Expr when err.failure_cases is False and err.data is not None"

patterns-established:
  - "Deferred computation: store nw.Expr, reconstruct failure_cases only when SchemaErrors is raised"
  - "nw.all_horizontal for lazy row-wise AND — works on polars LazyFrame and ibis nw.DataFrame without materializing"
  - "config_context override in backend validate() for drop_invalid_rows depth fix"

requirements-completed: [DIR-01, DIR-02, DIR-03, DIR-04, DIR-05, DIR-06, DIR-07]

# Metrics
duration: 180min
completed: 2026-03-24
---

# Phase 09 Plan 02: Implement drop_invalid_rows via nw.Expr Accumulation and nw.all_horizontal

**apply() now returns nw.Expr directly; drop_invalid_rows uses nw.all_horizontal on accumulated exprs — pure narwhals, no ibis delegation, no polars-specific LazyFrame slicing**

## Performance

- **Duration:** ~180 min
- **Started:** 2026-03-24
- **Completed:** 2026-03-24
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- All 20 previously-failing drop_invalid_rows tests now pass (polars lazy, polars nullable, ibis duckdb, ibis sqlite)
- Narwhals backend suite: 209 passed, 8 skipped, 1 xfailed — zero regressions from previous 208 passing
- drop_invalid_rows() is pure narwhals — no isinstance ibis check, no IbisSchemaBackend delegation, no polars-specific co[COK] indexing
- parity test test_drop_invalid_rows_expr_accumulation promoted from xfail(strict=True) to passing green test
- check_output stored on SchemaError is nw.Expr (not a wide table) — no wide table built during check loop

## Task Commits

1. **Task 1: Refactor apply() and add postprocess_expr_output() in checks.py** - `4eb784d` (feat)
2. **Task 2: Replace drop_invalid_rows() and update failure_cases_metadata() in base.py** - `2d8954a` (feat)

## Files Created/Modified

- `/Users/deepyaman/github/unionai-oss/pandera/pandera/backends/narwhals/checks.py` - apply() returns nw.Expr for native=False and element_wise; postprocess() dispatches to new postprocess_expr_output(); postprocess_expr_output() stores expr+deferred failure_cases
- `/Users/deepyaman/github/unionai-oss/pandera/pandera/backends/narwhals/base.py` - New drop_invalid_rows() with nw.all_horizontal accumulation; failure_cases_metadata() reconstructs failure_cases from nw.Expr; run_check() reconstructs failure_cases from nw.Expr when postprocess deferred None
- `/Users/deepyaman/github/unionai-oss/pandera/pandera/backends/narwhals/container.py` - config_context(SCHEMA_AND_DATA) when drop_invalid_rows=True to ensure data checks run on lazy frames
- `/Users/deepyaman/github/unionai-oss/pandera/tests/backends/narwhals/test_parity.py` - Promoted test from xfail to passing; redesigned to verify rows dropped (not SchemaErrors raised) plus separate nw.Expr contract check

## Decisions Made

- **config_context SCHEMA_AND_DATA for drop_invalid_rows:** polars LazyFrame defaults to SCHEMA_ONLY validation depth, which skips all @validate_scope(DATA) checks. When drop_invalid_rows=True, data checks must run to identify invalid rows. Fix: container.validate() wraps core_checks loop in config_context(SCHEMA_AND_DATA) when drop_invalid_rows=True.

- **ignore_na at column level after evaluation:** expr|expr.is_null() on an unevaluated nw.Expr causes ibis to produce incorrect SQL (IsNull on unevaluated expressions returns True for all rows due to SQL nullability semantics). Fix: evaluate expr to a single-column frame first via frame.select(expr.alias(KEY)), then apply is_null() on the concrete column values.

- **SERIES_CONTAINS_NULLS separate path in drop_invalid_rows:** check_nullable stores check_output as a wide LazyFrame with True=null (failing), not a nw.Expr. The new drop_invalid_rows handles this by reconstructing ~nw.col(selector).is_null() from err.schema.selector when reason_code==SERIES_CONTAINS_NULLS.

- **Test redesign for drop_invalid_rows_expr_accumulation:** The RED baseline test expected SchemaErrors to be raised with drop_invalid_rows=True — incorrect, the behavior is rows are silently dropped. Green test: (1) verify result has 2 rows after filtering -1, (2) verify check_output is nw.Expr using separate schema with drop_invalid_rows=False under config_context(SCHEMA_AND_DATA).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] container.validate() missing config_context override for drop_invalid_rows**
- **Found during:** Task 2 (testing drop_invalid_rows with polars LazyFrame)
- **Issue:** polars LazyFrame defaults to SCHEMA_ONLY validation depth in get_validation_depth(). With SCHEMA_ONLY, @validate_scope(DATA) on run_schema_component_checks returns CoreCheckResult(passed=True) without running checks. Result: error_handler.collected_errors is empty, drop_invalid_rows never called, all rows returned.
- **Fix:** Added config_context(validation_depth=SCHEMA_AND_DATA) wrapping the core_checks loop in container.validate() when drop_invalid_rows=True. The polars test conftest sets SCHEMA_AND_DATA globally, masking this bug in polars tests; narwhals parity tests have no conftest, so the bug was visible.
- **Files modified:** pandera/backends/narwhals/container.py
- **Verification:** test_drop_invalid_rows_expr_accumulation passes; polars drop_invalid_rows tests still pass (conftest overrides depth anyway)
- **Committed in:** 2d8954a (Task 2 commit)

**2. [Rule 1 - Bug] Test designed incorrectly for drop_invalid_rows=True behavior**
- **Found during:** Task 2 (running test_drop_invalid_rows_expr_accumulation)
- **Issue:** RED baseline test expected schema.validate(lf, lazy=True) with drop_invalid_rows=True to raise SchemaErrors. Actual behavior: invalid rows are silently dropped, no exception raised.
- **Fix:** Redesigned test to (1) assert validated result has 2 rows after row -1 is dropped, (2) use separate schema with drop_invalid_rows=False under config_context(SCHEMA_AND_DATA) to verify check_output is nw.Expr.
- **Files modified:** tests/backends/narwhals/test_parity.py
- **Verification:** test passes with correct assertions
- **Committed in:** 2d8954a (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (2 bugs)
**Impact on plan:** Both auto-fixes essential for correctness. The config_context fix is the core enabler — without it, drop_invalid_rows silently no-ops on lazy frames. The test redesign aligns the parity test with actual behavior. No scope creep.

## Issues Encountered

- **polars conftest masking bug:** The polars test conftest.py sets CONFIG.validation_depth=SCHEMA_AND_DATA for all polars tests, masking the fact that drop_invalid_rows with LazyFrame was silently no-oping. The narwhals parity tests have no such conftest, exposing the bug. Root cause traced via debug patching: error_handler.collected_errors was empty (collect_error never called) because @validate_scope(DATA) skipped all data checks under SCHEMA_ONLY depth.

- **ibis ignore_na bug:** expr|expr.is_null() on unevaluated nw.Expr in postprocess_expr_output() caused ibis to evaluate IsNull on the expression object itself (not its values), returning True for all rows. Fix: evaluate expr to single-column frame first, then apply is_null() on the concrete column — the check_col.with_columns(nw.col(KEY)|nw.col(KEY).is_null()) pattern.

## Next Phase Readiness

- Phase 09 complete — all drop_invalid_rows tests pass, narwhals backend suite has zero regressions
- 36 pre-existing failures in polars/ibis container tests are unrelated to this phase (regex, coerce, nested types, unique settings) — documented as pre-existing in deferred-items

---
*Phase: 09-accumulate-check-outputs-into-single-wide-table-for-narwhals-idiomatic-drop-invalid-rows*
*Completed: 2026-03-24*
