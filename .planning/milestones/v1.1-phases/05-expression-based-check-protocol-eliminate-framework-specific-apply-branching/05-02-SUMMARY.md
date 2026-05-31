---
phase: 05-expression-based-check-protocol-eliminate-framework-specific-apply-branching
plan: "02"
subsystem: narwhals
tags: [narwhals, checks, expressions, dispatcher, builtin-checks]

# Dependency graph
requires:
  - phase: 05-01
    provides: "RED baseline tests asserting nw.Expr dispatch for builtin checks and native=False user checks"
provides:
  - "All 14 builtin check functions rewritten with col_expr: nw.Expr as first arg, returning nw.Expr"
  - "Dispatcher._function_registry keyed on nw.Expr for all 14 builtins (was nw.LazyFrame)"
  - "No frame.select() calls remain in any builtin check function"
affects:
  - 05-03-PLAN

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Expression protocol: builtin checks accept col_expr: nw.Expr and return nw.Expr directly"
    - "Dispatcher auto-rekeys via get_first_arg_type reading the first-param annotation"

key-files:
  created: []
  modified:
    - pandera/backends/narwhals/builtin_checks.py

key-decisions:
  - "Transitional state accepted: apply() still calls fn(frame, key) so builtin_checks_pass/fail tests break (KeyError on nw.LazyFrame lookup); Plan 05-03 rewires apply() to complete the protocol"
  - "No frame.select() inside any builtin — return the expression directly so frame.with_columns(expr.alias(...)) works in Plan 05-03"

patterns-established:
  - "Pattern: builtin check = pure expression transform, no frame involvement"
  - "Pattern: col_expr: nw.Expr first arg causes Dispatcher to auto-key on nw.Expr via annotation reflection"

requirements-completed:
  - EXPR-02
  - EXPR-03

# Metrics
duration: 2min
completed: 2026-03-23
---

# Phase 5 Plan 02: Rewrite Builtin Checks to nw.Expr Protocol Summary

**All 14 narwhals builtin check functions rewritten from frame.select(nw.col(key) op value) to col_expr: nw.Expr -> nw.Expr, rekeying the Dispatcher on nw.Expr**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-23T06:08:26Z
- **Completed:** 2026-03-23T06:10:50Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Rewrote all 14 builtin check functions in `builtin_checks.py` to accept `col_expr: nw.Expr` as first arg and return `nw.Expr` directly
- Eliminated all `frame.select(...)` calls from builtin checks — pure expression transforms
- Dispatcher._function_registry now keyed on `nw.Expr` (confirmed via `assert nw.Expr in d._function_registry`)
- Docstrings updated: replaced `frame`/`key` param docs with `col_expr: narwhals column expression`

## Task Commits

Each task was committed atomically:

1. **Task 1: Rewrite all 14 builtin check functions to nw.Expr protocol** - `5157df0` (feat)

## Files Created/Modified
- `pandera/backends/narwhals/builtin_checks.py` — All 14 builtin functions rewritten: col_expr: nw.Expr first arg, nw.Expr return type, no frame.select() calls

## Decisions Made
- Accepted transitional failure state: apply() in checks.py still calls `check_fn(frame, key)` which KeyErrors against the new Dispatcher (keyed on nw.Expr, not nw.LazyFrame). This causes 58 additional test failures (test_builtin_checks_pass, test_builtin_checks_fail, test_apply_returns_wide_table). Plan 05-03 rewires apply() to resolve this.
- The ValueError guard in str_length (both min_value and max_value None) was preserved as a Python-level guard that fires before any expression is constructed.

## Deviations from Plan

None - plan executed exactly as written. The increased test failures (62 total vs 4 before) are the expected transitional state documented in the plan's done criteria.

## Issues Encountered
- None. The KeyError failures when apply() calls check_fn(frame, key) are the expected intermediate state — the plan explicitly documented that test_builtin_checks_pass/fail and test_builtin_check_routing remain FAILING until Plan 05-03 rewires apply().

## Next Phase Readiness
- All 14 builtins return nw.Expr — Plan 05-03 can now simplify apply() to `frame.with_columns(check_fn(nw.col(key)).alias(CHECK_OUTPUT_KEY))`
- The Dispatcher key change (nw.LazyFrame -> nw.Expr) is complete — Plan 05-03 must use `nw.Expr` in any registry lookups
- The ibis workaround block in apply() (lines 69-83 of checks.py) will be eliminated in Plan 05-03

---
*Phase: 05-expression-based-check-protocol-eliminate-framework-specific-apply-branching*
*Completed: 2026-03-23*
