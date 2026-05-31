---
phase: 05-expression-based-check-protocol-eliminate-framework-specific-apply-branching
plan: "01"
subsystem: testing
tags: [narwhals, checks, tdd, expressions, red-baseline]

# Dependency graph
requires: []
provides:
  - "RED baseline tests asserting nw.Expr dispatch for builtin checks and native=False user checks"
  - "test_builtin_check_routing patches _function_registry[nw.Expr] and asserts col_expr is nw.Expr"
  - "test_native_false_user_check defines user_check(col_expr) returning nw.Expr and asserts received nw.Expr"
affects:
  - 05-02-PLAN
  - 05-03-PLAN

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "TDD RED baseline: patch _function_registry[nw.Expr] (not [nw.LazyFrame]) to confirm dispatch target changes"
    - "Use .get() on registry to avoid KeyError during pre-migration RED phase"

key-files:
  created: []
  modified:
    - tests/backends/narwhals/test_checks.py

key-decisions:
  - "Use _function_registry.get(nw.Expr) instead of direct subscript to avoid KeyError before migration — test FAILs at assertion instead of erroring at setup"
  - "capturing_fn fallback (col_expr == 5) ensures postprocess doesn't crash during RED phase when original_fn is None"
  - "Teardown restores registry to pre-test state: pops nw.Expr key if it didn't exist before, restores original if it did"

patterns-established:
  - "Pattern: RED baseline for expression protocol — patch nw.Expr slot, run backend, assert isinstance(received, nw.Expr)"

requirements-completed:
  - EXPR-01
  - EXPR-06

# Metrics
duration: 2min
completed: 2026-03-23
---

# Phase 5 Plan 01: Expression-Based Check Protocol — RED Baseline Summary

**RED baseline tests asserting builtin checks receive nw.Expr (not frame+key), confirming what Plans 05-02 and 05-03 must implement to turn these GREEN**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-23T06:04:20Z
- **Completed:** 2026-03-23T06:06:00Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments
- Updated `test_builtin_check_routing` to patch `_function_registry[nw.Expr]` (was `nw.LazyFrame`) and assert the capturing function receives `nw.Expr` — fails with `assert 0 == 1` against current code
- Updated `test_native_false_user_check` to define `user_check(col_expr)` (single arg, returns `col_expr > 0`) and assert received value is `nw.Expr` — fails with `TypeError: takes 1 positional argument but 2 were given` against current code
- All 64 other tests in `test_checks.py` continue to pass (zero regressions)

## Task Commits

Each task was committed atomically:

1. **Task 1: Update test_builtin_check_routing to assert nw.Expr dispatch** - `a73672f` (test)
2. **Task 2: Update test_native_false_user_check to assert nw.Expr protocol** - `6e70705` (test)

_Note: TDD RED tasks — both tests intentionally fail against current code_

## Files Created/Modified
- `tests/backends/narwhals/test_checks.py` — Updated two tests to assert new expression protocol (nw.Expr-based calling convention)

## Decisions Made
- Used `_function_registry.get(nw.Expr)` instead of direct subscript to avoid `KeyError` in setup — the test should FAIL at the assertion (where it counts) not ERROR at setup
- Added a fallback expression in `capturing_fn` (`return col_expr == 5`) so postprocess doesn't crash if `original_fn` is None during RED phase
- Teardown correctly restores registry state: pops key if pre-existing was None, restores value otherwise

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Replaced direct registry subscript with .get() to prevent KeyError**
- **Found during:** Task 1 (test_builtin_check_routing update)
- **Issue:** Plan specified `original_fn = original_dispatcher._function_registry[nw.Expr]` but `nw.Expr` key doesn't exist in pre-migration code — causes `KeyError` in setup, making the test error rather than fail
- **Fix:** Used `.get(nw.Expr)` to return `None` safely; added fallback return in `capturing_fn`; teardown uses `.pop()` when original was None
- **Files modified:** tests/backends/narwhals/test_checks.py
- **Verification:** Test now fails with `assert 0 == 1` (FAILED, not ERROR) confirming correct RED baseline
- **Committed in:** a73672f (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 — bug: KeyError in test setup)
**Impact on plan:** Fix was necessary to achieve a proper FAIL vs ERROR baseline. Plan's done criterion requires FAIL not ERROR.

## Issues Encountered
- Plan specified direct `_function_registry[nw.Expr]` subscript which would KeyError before Phase 5 migrates the registry — corrected to `.get()` to maintain proper RED/FAIL semantics

## Next Phase Readiness
- RED baseline established for both routing tests
- Plans 05-02 (update builtin check signatures) and 05-03 (rewrite apply()) can now proceed
- These 4 test failures (2 tests x 2 backends) will turn GREEN after 05-02 + 05-03 complete

---
*Phase: 05-expression-based-check-protocol-eliminate-framework-specific-apply-branching*
*Completed: 2026-03-23*
