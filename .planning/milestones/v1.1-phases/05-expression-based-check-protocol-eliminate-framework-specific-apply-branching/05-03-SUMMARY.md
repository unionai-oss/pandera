---
phase: 05-expression-based-check-protocol-eliminate-framework-specific-apply-branching
plan: "03"
subsystem: narwhals
tags: [narwhals, checks, expressions, ibis, polars, apply]

# Dependency graph
requires:
  - phase: 05-01
    provides: "RED baseline tests asserting nw.Expr dispatch for builtin checks and native=False user checks"
  - phase: 05-02
    provides: "All 14 builtin check functions rewritten with col_expr: nw.Expr as first arg, returning nw.Expr"
provides:
  - "apply() rewritten using uniform expression protocol — no ibis row_number join, no Dispatcher workaround"
  - "element_wise branch: map_batches expr → frame.with_columns(expr.alias(CHECK_OUTPUT_KEY))"
  - "native=False branch: check_fn(nw.col(key)) → frame.with_columns(expr.alias(CHECK_OUTPUT_KEY))"
  - "native=True branch: unchanged — to_native → check_fn → _normalize_native_output (now returns wide table for ir.BooleanColumn)"
  - "All Phase 5 tests GREEN — 68 pass, 0 failures in test_checks.py"
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Uniform expression protocol: all non-native branches return frame.with_columns(expr.alias(CHECK_OUTPUT_KEY))"
    - "ibis.Table.mutate(**{CHECK_OUTPUT_KEY: bool_column}) attaches BooleanColumn as wide table column"

key-files:
  created: []
  modified:
    - pandera/backends/narwhals/checks.py

key-decisions:
  - "element_wise try/except wraps frame.with_columns() call, not just map_batches() — NotImplementedError fires at evaluation time in narwhals for SQL-lazy backends"
  - "_normalize_native_output ir.BooleanColumn branch uses native.mutate(**{CHECK_OUTPUT_KEY: out}) to produce wide table — old native.select() approach produced 1-column frame causing postprocess failure_cases.select(key) to fail"
  - "4 remaining narwhals suite failures are pre-existing (ibis DatabaseTable vs Table naming, pyarrow.Table vs ibis.Table native type assertions) — not caused by Phase 5"

patterns-established:
  - "Pattern: apply() is ~30 lines with three clean branches — no backend detection, no row_number join, no reassembly block"
  - "Pattern: _normalize_native_output for ir.BooleanColumn produces wide table via mutate() — consistent with with_columns() approach in other branches"

requirements-completed:
  - EXPR-01
  - EXPR-04
  - EXPR-05
  - EXPR-06
  - EXPR-07

# Metrics
duration: 15min
completed: 2026-03-23
---

# Phase 5 Plan 03: Rewrite apply() Using Uniform Expression Protocol Summary

**apply() reduced from ~100 lines (with ibis row_number join) to ~30 lines using frame.with_columns(expr.alias(CHECK_OUTPUT_KEY)) for all three branches — all 68 test_checks.py tests GREEN**

## Performance

- **Duration:** 15 min
- **Started:** 2026-03-23T06:10:50Z
- **Completed:** 2026-03-23T06:25:00Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments
- Deleted ibis row_number join block (~40 lines) from apply()
- Deleted Dispatcher workaround (isinstance check, nw.LazyFrame registry lookup, partial kwargs)
- Deleted rename/all_horizontal reassembly block
- element_wise branch now: `expr = selector.map_batches(...); return frame.with_columns(expr.alias(CHECK_OUTPUT_KEY))`
- native=False branch now: `expr = check_fn(nw.col(key)); return frame.with_columns(expr.alias(CHECK_OUTPUT_KEY))`
- Fixed `_normalize_native_output` ir.BooleanColumn path: now uses `native.mutate(**{CHECK_OUTPUT_KEY: out})` for wide table instead of narrow 1-column `native.select(out.name(...))`
- Removed unused `_materialize` import from top of file
- All 68 tests in test_checks.py pass (was 62 failures before this plan)
- Full narwhals suite: 4 failures all pre-existing, no regressions

## Task Commits

Each task was committed atomically:

1. **Task 1: Rewrite apply() — delete ibis join, Dispatcher workaround, reassembly block** - `e22a4b7` (feat)
2. **Task 2: Run full narwhals backend suite and confirm no regressions** - `e725b5f` (fix)

## Files Created/Modified
- `pandera/backends/narwhals/checks.py` — apply() rewritten (19 insertions, 87 deletions); _normalize_native_output ir.BooleanColumn fixed to produce wide table; _materialize import removed

## Decisions Made
- The try/except NotImplementedError in element_wise must wrap the `frame.with_columns()` call, not just the `map_batches()` call — narwhals raises NotImplementedError at expression evaluation time (during with_columns), not at expression construction time
- `_normalize_native_output` for ir.BooleanColumn was producing a 1-column frame via `native.select(out.name(CHECK_OUTPUT_KEY))`. This worked with the old code because the reassembly block (now deleted) attached it to the original frame. Fixed to use `native.mutate(**{CHECK_OUTPUT_KEY: out})` which produces a wide table directly — consistent with how the native=False and element_wise branches work

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed element_wise try/except scope**
- **Found during:** Task 1 (rewrite apply())
- **Issue:** `selector.map_batches(...)` returns an Expr lazily — narwhals raises NotImplementedError at `frame.with_columns(expr)` evaluation, not at map_batches construction. The try/except only wrapped map_batches, so the NotImplementedError was uncaught and showed narwhals' raw message, causing test_element_wise_sql_lazy_raises to fail.
- **Fix:** Moved `return frame.with_columns(expr.alias(CHECK_OUTPUT_KEY))` inside the try block so both expression construction and evaluation are covered.
- **Files modified:** pandera/backends/narwhals/checks.py
- **Verification:** test_element_wise_sql_lazy_raises[ibis] passes (was failing with mismatched error message)
- **Committed in:** e22a4b7 (Task 1 commit)

**2. [Rule 1 - Bug] Fixed _normalize_native_output ir.BooleanColumn producing narrow table**
- **Found during:** Task 2 (full narwhals suite)
- **Issue:** With the old ibis row_number join removed, the native=True path for ir.BooleanColumn now returns directly from _normalize_native_output. The old code produced a 1-column frame (only CHECK_OUTPUT_KEY), causing postprocess_lazyframe_output's `failure_cases.select(check_obj.key)` to fail with ColumnNotFoundError.
- **Fix:** Changed `native.select(out.name(CHECK_OUTPUT_KEY))` to `native.mutate(**{CHECK_OUTPUT_KEY: out})` so the wide table (original columns + CHECK_OUTPUT_KEY) is returned directly.
- **Files modified:** pandera/backends/narwhals/checks.py
- **Verification:** test_custom_boolean_column_check_passes[ibis] passes; full suite has no new failures
- **Committed in:** e725b5f (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (both Rule 1 - Bug)
**Impact on plan:** Both fixes were necessary for correctness. The first was an expression evaluation timing issue. The second was a consequence of removing the old reassembly block that _normalize_native_output had previously relied on. No scope creep.

## Issues Encountered
- narwhals raises NotImplementedError at expression evaluation, not construction — required moving the return statement into the try/except block
- _normalize_native_output's ir.BooleanColumn branch was implicitly dependent on the old reassembly block (ibis row_number join) to produce a wide table. Removing the join exposed this dependency and required fixing mutate to produce the wide table directly.

## Next Phase Readiness
- Phase 5 complete — all 5 requirements (EXPR-01, EXPR-04, EXPR-05, EXPR-06, EXPR-07) satisfied
- apply() is now backend-agnostic: no isinstance checks, no ibis imports, no polars-specific code
- 4 pre-existing failures documented: test_greater_than_fails_failure_cases_{type,values}, test_custom_check_receives_table_and_key (DatabaseTable vs Table ibis naming), test_failure_cases_native_ibis (pyarrow vs ibis.Table type)

## Self-Check: PASSED
- pandera/backends/narwhals/checks.py: EXISTS
- Commits e22a4b7 and e725b5f: both present in git log

---
*Phase: 05-expression-based-check-protocol-eliminate-framework-specific-apply-branching*
*Completed: 2026-03-23*
