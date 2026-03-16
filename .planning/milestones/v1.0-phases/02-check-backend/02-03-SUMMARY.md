---
phase: 02-check-backend
plan: 03
subsystem: checks
tags: [narwhals, builtin-checks, dispatcher, ibis, polars]

# Dependency graph
requires:
  - phase: 02-check-backend-01
    provides: NarwhalsCheckBackend base implementation and conftest fixtures
  - phase: 02-check-backend-02
    provides: test stubs for CHECKS-02 with xfail markers
provides:
  - 14 narwhals builtin check registrations in builtin_checks.py via @register_builtin_check
  - Dispatcher-based routing fix in NarwhalsCheckBackend.apply()
  - SQL-lazy (ibis) materialization support in NarwhalsCheckBackend
affects: [03-schema-backend, 04-validation-runner]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Use Python comparison operators (==, !=, >, >=, <, <=) on narwhals Expr (not .eq/.ne/.gt etc.)
    - Use is_between(closed=) for range checks mapping include_min/include_max
    - Use ~expr tilde for NOT IS IN checks (not .not_())
    - Dispatcher registry lookup (NarwhalsData in dispatcher._function_registry) to detect builtin checks
    - SQL-lazy materialization via nw.to_native().execute() for ibis backends

key-files:
  created:
    - pandera/backends/narwhals/builtin_checks.py
  modified:
    - pandera/backends/narwhals/checks.py
    - tests/backends/narwhals/test_checks.py

key-decisions:
  - "narwhals Expr has no .eq/.ne/.gt/.ge/.lt/.le — Python comparison operators (==, !=, >, etc.) are used instead"
  - "NarwhalsCheckBackend.apply() detects builtins via Dispatcher._function_registry[NarwhalsData] lookup, not signature inspection"
  - "ibis backend returns nw.DataFrame (not nw.LazyFrame) from .select(); materialization uses nw.to_native().execute()"
  - "test_checks assertions use == instead of is to handle np.bool_ from ibis"

patterns-established:
  - "Builtin check detection: isinstance(inner_fn, Dispatcher) and NarwhalsData in inner_fn._function_registry"
  - "_materialize() helper handles both nw.LazyFrame (.collect()) and SQL-lazy nw.DataFrame (nw.to_native().execute())"

requirements-completed: [CHECKS-02]

# Metrics
duration: 6min
completed: 2026-03-10
---

# Phase 02 Plan 03: Narwhals Builtin Checks Summary

**14 narwhals builtin checks registered via @register_builtin_check with NarwhalsData dispatch, supporting both Polars and Ibis backends through Python comparison operators and SQL-lazy materialization**

## Performance

- **Duration:** 6 min
- **Started:** 2026-03-10T02:01:25Z
- **Completed:** 2026-03-10T02:07:35Z
- **Tasks:** 1
- **Files modified:** 3 (1 created)

## Accomplishments
- Implemented all 14 narwhals builtin checks (equal_to through str_length) in `pandera/backends/narwhals/builtin_checks.py`
- Fixed `NarwhalsCheckBackend.apply()` routing to detect Dispatcher-based builtins via registry lookup
- Fixed SQL-lazy (ibis) materialization in `postprocess_lazyframe_output` via `_materialize()` helper
- All 56 CHECKS-02 test cases pass for both polars and ibis backends

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement all 14 narwhals builtin checks** - `de7a119` (feat)

**Plan metadata:** (docs: complete plan — committed after SUMMARY)

## Files Created/Modified
- `pandera/backends/narwhals/builtin_checks.py` - 14 builtin check registrations with NarwhalsData first-arg typing
- `pandera/backends/narwhals/checks.py` - Dispatcher routing fix + ibis materialization support
- `tests/backends/narwhals/test_checks.py` - Removed xfail markers, fixed check_passed extraction for DataFrame/LazyFrame/bool

## Decisions Made
- narwhals Expr uses Python comparison operators (`==`, `!=`, `>`, `>=`, `<`, `<=`) — the `.eq/.ne/.gt/.ge/.lt/.le` methods noted in the plan context don't exist on narwhals Expr in stable.v1
- Dispatcher._function_registry lookup is the correct way to detect builtin narwhals checks (not inspect.signature which gives `*args` for Dispatcher)
- ibis backend wraps tables as `nw.DataFrame` (not `nw.LazyFrame`), requiring `nw.to_native().execute()` for materialization instead of `.collect()`
- Test assertions updated to use `== True/False` instead of `is True/is False` because ibis returns `np.bool_` not Python `bool`

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] narwhals Expr lacks .eq/.ne/.gt/.ge/.lt/.le methods**
- **Found during:** Task 1 (implementation)
- **Issue:** Plan specified `.eq(value)`, `.ne(value)`, `.gt()`, `.ge()`, `.lt()`, `.le()` but narwhals stable.v1 Expr only supports Python comparison operators
- **Fix:** Replaced all `.eq/.ne/.gt/.ge/.lt/.le` calls with `==`, `!=`, `>`, `>=`, `<`, `<=` operators
- **Files modified:** pandera/backends/narwhals/builtin_checks.py
- **Verification:** Tests pass after fix
- **Committed in:** de7a119 (Task 1 commit)

**2. [Rule 1 - Bug] NarwhalsCheckBackend.apply() routing miss for Dispatcher-based builtins**
- **Found during:** Task 1 (test execution)
- **Issue:** `inspect.signature(Dispatcher)` returns `*args` with no type annotation, so `first_param.annotation is NarwhalsData` was always False, routing all builtins to user-defined path and failing with `KeyError: PolarsLazyFrame`
- **Fix:** Added Dispatcher instance check with `NarwhalsData in inner_fn._function_registry` for accurate builtin detection
- **Files modified:** pandera/backends/narwhals/checks.py
- **Verification:** All 28 polars builtin tests pass after fix
- **Committed in:** de7a119 (Task 1 commit)

**3. [Rule 1 - Bug] ibis backend returns nw.DataFrame, not nw.LazyFrame**
- **Found during:** Task 1 (ibis test execution)
- **Issue:** `postprocess_lazyframe_output` called `.collect()` on check output, but ibis frames are `nw.DataFrame` (not `nw.LazyFrame`) and have no `.collect()` — raises `AttributeError`
- **Fix:** Added `_materialize()` static method that branches on `isinstance(frame, nw.LazyFrame)` vs SQL-lazy (uses `nw.to_native().execute()`)
- **Files modified:** pandera/backends/narwhals/checks.py
- **Verification:** All 28 ibis builtin tests pass after fix
- **Committed in:** de7a119 (Task 1 commit)

**4. [Rule 1 - Bug] Test assertions used `is True/False` but ibis returns np.bool_**
- **Found during:** Task 1 (ibis fail tests)
- **Issue:** `assert val is False` failed because ibis produces `np.False_` not Python `False`; also test's `hasattr(passed, "collect")` doesn't handle `nw.DataFrame`
- **Fix:** Updated both test functions to use `isinstance(passed, nw.LazyFrame/nw.DataFrame)` for extraction and `== True/False` for assertion
- **Files modified:** tests/backends/narwhals/test_checks.py
- **Verification:** All 56 builtin tests pass after fix
- **Committed in:** de7a119 (Task 1 commit)

---

**Total deviations:** 4 auto-fixed (all Rule 1 bugs — API mismatch between plan documentation and actual narwhals/ibis behavior)
**Impact on plan:** All auto-fixes were required for correctness. Core implementation logic (14 check functions, in_range mapping, notin tilde, str_matches anchoring, str_length cases) matches plan exactly.

## Issues Encountered
- narwhals stable.v1 Expr comparison API differs from Polars `.eq/.ne` naming — uses Python dunder methods instead
- ibis narwhals integration uses nw.DataFrame (not LazyFrame) because ibis expressions are already "lazy" in a SQL sense but narwhals doesn't wrap them as LazyFrame

## Next Phase Readiness
- CHECKS-02 complete: all 14 builtin checks registered and working for both Polars and Ibis
- CHECK_FUNCTION_REGISTRY Dispatcher correctly routes NarwhalsData to narwhals implementations
- NarwhalsCheckBackend handles both eager (Polars) and SQL-lazy (Ibis) backends correctly
- Ready for schema backend phase

## Self-Check: PASSED
- FOUND: pandera/backends/narwhals/builtin_checks.py
- FOUND: commit de7a119

---
*Phase: 02-check-backend*
*Completed: 2026-03-10*
