---
phase: 01-foundation
plan: 02
subsystem: infra
tags: [narwhals, narwhals.stable.v1, dtype-engine, coercion, pytest]

# Dependency graph
requires:
  - phase: 01-01
    provides: NarwhalsData, NarwhalsCheckResult, _to_native, test scaffold xfail stubs
provides:
  - pandera/engines/narwhals_engine.py with DataType base, Engine metaclass, 18 dtype registrations
  - coerce() returning nw.LazyFrame (lazy, no collect)
  - try_coerce() raising ParserError with native failure_cases
  - Engine.dtype() resolving narwhals dtype classes and instances
  - from_parametrized_dtype dispatch for DateTime and Duration
affects:
  - 01-03 (schema model — uses Engine.dtype() to resolve column dtypes)
  - 01-04 (engine registration — opt-in path, narwhals_engine already structured for it)
  - Phase 2 (check backend — imports Engine and DataType from narwhals_engine)
  - Phase 3 (column backend — resolves column dtypes through this engine)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Narwhals coercion uses lf.with_columns(nw.col(key).cast(dtype)) — no frame-level .cast()
    - try_coerce() calls .collect() to force lazy evaluation; catches COERCION_ERRORS
    - Lazy imports inside coerce/try_coerce prevent circular imports with pandera.api.narwhals
    - from_parametrized_dtype return type annotated as Any to avoid forward-ref NameError
    - narwhals_engine.py never imported from any __init__.py — strict opt-in design

key-files:
  created:
    - pandera/engines/narwhals_engine.py
  modified:
    - tests/backends/narwhals/test_narwhals_dtypes.py

key-decisions:
  - "Lazy imports of NarwhalsData and _to_native inside coerce/try_coerce prevent circular imports"
  - "from_parametrized_dtype return type is Any (not forward ref string) to avoid NameError in get_type_hints"
  - "try_coerce() uses _to_native(frame.collect()) for native failure_cases — collected before passing to ParserError"
  - "narwhals_engine.py NOT imported from any __init__.py — maintained strict opt-in isolation"

patterns-established:
  - "narwhals cast pattern: lf.with_columns(nw.col(key).cast(self.type)) for single-column, nw.all().cast() for wildcard"
  - "Engine.dtype() delegates to engine.Engine.dtype() and raises clear TypeError on miss"
  - "Parameterized dtypes (DateTime, Duration, List) use @immutable(init=True) with from_parametrized_dtype classmethod"

requirements-completed: [ENGINE-01, ENGINE-02, ENGINE-03]

# Metrics
duration: 5min
completed: 2026-03-09
---

# Phase 1 Plan 02: Narwhals Dtype Engine Summary

**narwhals dtype engine with 18 registered types (Int/UInt/Float/String/Bool/Date/DateTime/Duration/Categorical/List/Struct), lazy coerce(), try_coerce() with native ParserError failure_cases, and from_parametrized_dtype dispatch for parameterized types**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-09T22:52:02Z
- **Completed:** 2026-03-09T22:57:02Z
- **Tasks:** 1
- **Files modified:** 2

## Accomplishments

- Created `pandera/engines/narwhals_engine.py` with DataType base class and Engine metaclass using narwhals.stable.v1
- Registered 18 narwhals dtypes: Int8/16/32/64, UInt8/16/32/64, Float32/64, String, Bool, Date, DateTime, Duration, Categorical, List, Struct
- `coerce()` uses `lf.with_columns(nw.col(key).cast(self.type))` — fully lazy, no `.collect()` call
- `try_coerce()` forces evaluation via `.collect()`, catches `COERCION_ERRORS`, raises `ParserError` with native (non-narwhals) `failure_cases`
- `DateTime` and `Duration` support parameterized instantiation and `from_parametrized_dtype` dispatch
- `List` supports optional `inner` type with `from_parametrized_dtype`
- Removed 22 xfail markers from test scaffold — all 26 tests now pass GREEN

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement narwhals dtype engine** - `fc80e9e` (feat)

**Plan metadata:** (docs commit to follow)

_Note: TDD task — implementation flips all 22 ENGINE xfail stubs to GREEN_

## Files Created/Modified

- `pandera/engines/narwhals_engine.py` - Complete narwhals dtype engine: DataType base, Engine metaclass, 18 dtype registrations, coerce/try_coerce
- `tests/backends/narwhals/test_narwhals_dtypes.py` - Removed xfail markers from 22 ENGINE tests; all 26 tests pass GREEN

## Decisions Made

- Used lazy imports inside `coerce()` and `try_coerce()` for `NarwhalsData` and `_to_native` to avoid circular imports between `pandera.engines.narwhals_engine` and `pandera.api.narwhals`
- Annotated `from_parametrized_dtype` return type as `Any` instead of a string forward reference to avoid `NameError` in `get_type_hints()` during engine registration
- `try_coerce()` collects the original (pre-coercion) frame as `failure_cases` since narwhals doesn't provide row-level coercibility metadata (simpler and correct for Phase 1)
- `narwhals_engine.py` kept completely out of all `__init__.py` files — opt-in isolation maintained

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed forward-reference NameError in from_parametrized_dtype**
- **Found during:** Task 1 (after initial implementation attempt)
- **Issue:** `from_parametrized_dtype(cls, ...) -> "DateTime"` caused `NameError: name 'DateTime' is not defined` in `get_type_hints()` during `Engine.register_dtype()` decoration
- **Fix:** Changed return type annotation from `"DateTime"` / `"Duration"` / `"List"` to `Any`
- **Files modified:** `pandera/engines/narwhals_engine.py`
- **Verification:** Engine registration succeeds, all 26 tests pass
- **Committed in:** `fc80e9e` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Fix was necessary for engine registration to work at all. No scope creep.

## Issues Encountered

- narwhals `from_parametrized_dtype` return type annotation with forward reference caused `NameError` in `get_type_hints()` during engine decorator registration — fixed by using `Any` return type annotation.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `Engine.dtype()` is fully functional and resolves all 18 narwhals dtypes
- `coerce()` and `try_coerce()` are ready for Phase 2's check backend to call
- `narwhals_engine.py` is importable directly — Phase 4 will wire up the opt-in activation path
- No side effects: importing `pandera.engines.narwhals_engine` does not affect any other engine registrations

---
*Phase: 01-foundation*
*Completed: 2026-03-09*

## Self-Check: PASSED

- `pandera/engines/narwhals_engine.py` — FOUND
- `tests/backends/narwhals/test_narwhals_dtypes.py` — FOUND (modified)
- Commit `fc80e9e` — FOUND
- `python -m pytest tests/backends/narwhals/ -q` — 26 passed, 0 failed
