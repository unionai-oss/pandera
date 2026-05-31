---
phase: 03-fix-ibischeckbackend-delegation-via-apply-type-dispatch
plan: "02"
subsystem: api
tags: [narwhals, checks, check-backend, ibis, polars, dispatch]

# Dependency graph
requires:
  - phase: 03-fix-ibischeckbackend-delegation-via-apply-type-dispatch
    plan: "01"
    provides: "Check.native flag; all 14 builtin check signatures refactored to (frame, key, ...)"
provides:
  - "NarwhalsCheckBackend.apply() with three explicit branches: element_wise / native=True / native=False"
  - "_normalize_native_output static method normalizing ibis BooleanScalar/BooleanColumn/Table outputs"
  - "Simplified __call__() with no ibis delegation block"
  - "All 14 builtin checks pass on both Polars and Ibis data"
  - "New tests for native=True polars/ibis dispatch and native=False convention"
affects:
  - "03-03: phase final verification"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "apply() dispatches on self.check.native flag — three explicit branches, no type-detection magic"
    - "_normalize_native_output guards ibis import via try/except ImportError for optional dependency"
    - "native=False path with Dispatcher: look up nw.LazyFrame impl directly for nw.DataFrame frames (ibis)"
    - "postprocess_bool_output: polars fallback for SQL-lazy (ibis) backends that can't use from_dict"

key-files:
  created: []
  modified:
    - pandera/backends/narwhals/checks.py
    - tests/backends/narwhals/test_checks.py

key-decisions:
  - "Dispatcher used in native=False branch to handle ibis nw.DataFrame frames: look up nw.LazyFrame impl and call with kwargs directly — avoids KeyError on nw.DataFrame type"
  - "postprocess_bool_output falls back to polars LazyFrame when nw.from_dict fails for ibis (SQL-lazy) backends"
  - "test_builtin_check_routing updated to patch Dispatcher registry directly to capture what builtin receives — avoids wrapping check in a way that loses native=False flag"

patterns-established:
  - "NarwhalsCheckBackend.apply(): native=True -> nw.to_native() + check_fn(native_frame, key); native=False -> check_fn(nw_frame, key)"
  - "_normalize_native_output: try: import ibis; handle BooleanScalar/BooleanColumn/Table; except ImportError: pass; return out"

requirements-completed: []

# Metrics
duration: 5min
completed: 2026-03-22
---

# Phase 03 Plan 02: apply() Rewrite and native=True Dispatch Convention Summary

**NarwhalsCheckBackend.apply() rewritten with three explicit branches (element_wise/native=True/native=False), IbisCheckBackend delegation block removed from __call__(), and all 14 builtin checks now pass on both Polars and Ibis data**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-22T16:55:58Z
- **Completed:** 2026-03-22T16:61:05Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Rewrote `apply()` to dispatch purely on `self.check.native` — three explicit branches replacing Dispatcher/inspect detection
- Added `_normalize_native_output` static method handling ibis BooleanScalar, BooleanColumn, and Table outputs
- Simplified `__call__()` to four lines — no ibis delegation block, no `IbisCheckBackend` call
- Removed `import inspect` and top-level `Dispatcher` usage from `checks.py`
- All 14 builtin checks now pass on both Polars and Ibis data (66 tests pass, 4 skip appropriately)
- Updated `test_builtin_check_routing` (removed xfail, new assertion)
- Added 4 new tests: native=True polars, native=True ibis, native=False, ibis scalar normalization

## Task Commits

Each task was committed atomically:

1. **Task 1: Rewrite apply(), add _normalize_native_output, simplify __call__** - `b43548e` (feat)
2. **Task 2: Update and add tests for native=True dispatch convention** - `0c358d0` (test)

## Files Created/Modified
- `pandera/backends/narwhals/checks.py` - apply() rewritten, _normalize_native_output added, __call__ simplified, postprocess_bool_output ibis-safe, native=False Dispatcher fix for ibis
- `tests/backends/narwhals/test_checks.py` - test_builtin_check_routing updated, test_user_defined_check_routing updated, 4 new tests added

## Decisions Made
- `Dispatcher` still used inside `native=False` branch (not dead) — needed to bypass type-dispatch for ibis frames that arrive as `nw.DataFrame` instead of `nw.LazyFrame`. The `nw.LazyFrame` implementation is looked up and called directly with the partial's kwargs.
- `postprocess_bool_output` uses try/except to fall back to polars `LazyFrame` when `nw.from_dict()` fails for ibis namespace (SQL-lazy, no eager from_dict support).
- `test_builtin_check_routing` patches the Dispatcher registry directly rather than wrapping the check — wrapping would create a new Check with `native=True` (default), losing the `native=False` flag that builtins need.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed ibis nw.DataFrame KeyError in native=False Dispatcher path**
- **Found during:** Task 1 (apply() rewrite)
- **Issue:** Ibis frames wrapped in narwhals become `nw.DataFrame` (not `nw.LazyFrame`), but the builtin Dispatcher is keyed on `nw.LazyFrame`. Calling `check_fn(ibis_frame, key)` raised `KeyError: <class 'narwhals.stable.v1.DataFrame'>`.
- **Fix:** For `native=False` path, detect if inner fn is a Dispatcher and frame is not `nw.LazyFrame`; look up `nw.LazyFrame` registered implementation directly and call with extracted kwargs.
- **Files modified:** pandera/backends/narwhals/checks.py
- **Verification:** All 14 ibis builtin check tests pass
- **Committed in:** b43548e (Task 1 commit)

**2. [Rule 1 - Bug] Fixed postprocess_bool_output for ibis (SQL-lazy) backends**
- **Found during:** Task 1 (test_user_defined_check_routing[ibis] failure)
- **Issue:** `nw.from_dict({key: [True]}, native_namespace=ibis_ns)` raises `ValueError: ibis support in Narwhals is lazy-only, but from_dict is an eager-only function`.
- **Fix:** Added try/except in `postprocess_bool_output`; falls back to polars `LazyFrame` for bool scalar results when the frame's native namespace doesn't support `from_dict`.
- **Files modified:** pandera/backends/narwhals/checks.py
- **Verification:** test_user_defined_check_routing[ibis] passes; test_ibis_boolean_scalar_normalization passes
- **Committed in:** b43548e (Task 1 commit)

---

**Total deviations:** 2 auto-fixed (2 Rule 1 bugs)
**Impact on plan:** Both bugs were directly caused by the apply() rewrite — pre-existing code never hit these paths. All fixes required for correctness. No scope creep.

## Issues Encountered
- `test_builtin_check_routing` needed to patch the Dispatcher registry directly (not wrap the check fn) because wrapping creates a new Check with `native=True` default, breaking the `native=False` routing being tested.

## Next Phase Readiness
- Phase goal achieved: `IbisCheckBackend` is no longer delegated to from `NarwhalsCheckBackend`
- All checks (builtin and user-defined, polars and ibis) follow a single explicit dispatch path via `self.check.native`
- Ready for phase 03-03 (final verification if any)

## Self-Check: PASSED

All files exist and commits are verified:
- `pandera/backends/narwhals/checks.py` - FOUND
- `tests/backends/narwhals/test_checks.py` - FOUND
- `.planning/phases/03-fix-ibischeckbackend-delegation-via-apply-type-dispatch/03-02-SUMMARY.md` - FOUND
- Commit `b43548e` (Task 1) - FOUND
- Commit `0c358d0` (Task 2) - FOUND

---
*Phase: 03-fix-ibischeckbackend-delegation-via-apply-type-dispatch*
*Completed: 2026-03-22*
