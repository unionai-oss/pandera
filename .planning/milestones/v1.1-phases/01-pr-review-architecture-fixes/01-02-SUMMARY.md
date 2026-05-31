---
phase: 01-pr-review-architecture-fixes
plan: 02
subsystem: api
tags: [narwhals, error-handler, polars, ibis, backends]

# Dependency graph
requires:
  - phase: 01-pr-review-architecture-fixes/01-01
    provides: NarwhalsErrorHandler subclass in pandera/api/narwhals/error_handler.py
provides:
  - NarwhalsErrorHandler wired into container.py, components.py, and base.py
  - container.py is backend-agnostic (no polars-specific type checks)
  - Accurate comments in run_schema_component_checks
  - Narwhals capitalization corrected across all narwhals backend files
affects:
  - 01-03
  - 01-04

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Duck-typing via hasattr(return_type, 'collect') to detect lazy vs eager return types
    - Dynamic Column class detection via schema.__class__.__module__ for backend-agnostic dispatch
    - NarwhalsErrorHandler imported from pandera.api.narwhals.error_handler in all narwhals backends

key-files:
  created: []
  modified:
    - pandera/backends/narwhals/container.py
    - pandera/backends/narwhals/components.py
    - pandera/backends/narwhals/base.py
    - pandera/backends/narwhals/checks.py
    - pandera/backends/narwhals/builtin_checks.py
    - pandera/backends/narwhals/register.py

key-decisions:
  - "hasattr(return_type, 'collect') on the class (not instance) correctly distinguishes lazy (pl.LazyFrame) from eager (pl.DataFrame/ibis.Table) return types without importing polars"
  - "Dynamic Column import via schema.__class__.__module__ check avoids hardcoding polars in a backend-agnostic method"

patterns-established:
  - "Backend-agnostic duck-typing: check class attributes, not isinstance with polars types"
  - "All narwhals backend files import ErrorHandler from pandera.api.narwhals.error_handler"

requirements-completed:
  - ARCH-01
  - ARCH-02
  - ARCH-03
  - ARCH-04

# Metrics
duration: 5min
completed: 2026-03-21
---

# Phase 01 Plan 02: Wire NarwhalsErrorHandler and fix polars coupling Summary

**NarwhalsErrorHandler wired into all three narwhals backends; container.py decoupled from polars-specific imports using duck-typing and dynamic schema module detection**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-03-21T22:50:35Z
- **Completed:** 2026-03-21T22:55:06Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments

- Replaced `from pandera.api.base.error_handler import ErrorHandler` with `NarwhalsErrorHandler` in container.py, components.py, and base.py
- Removed `import polars as pl` from container.py — no remaining polars-specific usage
- Fixed `_to_frame_kind_nw` to use duck-typing on `return_type` class instead of `issubclass(return_type, pl.DataFrame)`
- Fixed `collect_schema_components` to dynamically detect Column class via `schema.__class__.__module__`
- Updated misleading comment in `run_schema_component_checks`
- Corrected "narwhals" -> "Narwhals" capitalization across all 6 narwhals backend files in docstrings/comments

## Task Commits

Each task was committed atomically:

1. **Task 1: Fix container.py — polars imports, _to_frame_kind_nw, collect_schema_components, comment** - `9f9e38b` (fix)
2. **Task 2: Wire NarwhalsErrorHandler into components.py and base.py, fix capitalization** - `b01eb85` (fix)

**Plan metadata:** (see final commit)

## Files Created/Modified

- `pandera/backends/narwhals/container.py` - NarwhalsErrorHandler, duck-typed return type detection, dynamic Column import, updated comment, removed polars import
- `pandera/backends/narwhals/components.py` - NarwhalsErrorHandler, capitalization fixes
- `pandera/backends/narwhals/base.py` - NarwhalsErrorHandler, capitalization fixes
- `pandera/backends/narwhals/checks.py` - Capitalization fixes in docstrings/comments
- `pandera/backends/narwhals/builtin_checks.py` - Module docstring capitalization fix
- `pandera/backends/narwhals/register.py` - Module docstring capitalization fixes

## Decisions Made

- `hasattr(return_type, "collect")` on the class object (not the instance) distinguishes lazy types (pl.LazyFrame has a `.collect` classmethod) from eager types — avoids importing polars while being correct for all supported backends
- Dynamic Column detection via `schema.__class__.__module__` is sufficient for ibis vs polars discrimination without a more complex registry lookup

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed _to_frame_kind_nw regression: hasattr(native, "collect") was checking the wrong object**
- **Found during:** Task 1 (_to_frame_kind_nw fix)
- **Issue:** Plan specified `hasattr(native, "collect")` on the native instance — but a Polars LazyFrame (the native result) always has `.collect()`, causing pl.LazyFrame inputs to be incorrectly materialized to pl.DataFrame on return
- **Fix:** Changed to `hasattr(return_type, "collect")` on the return_type *class*, which distinguishes lazy types (pl.LazyFrame class has `collect`) from eager types (pl.DataFrame class does not). Test `test_validate_polars_lazyframe` verified the fix.
- **Files modified:** pandera/backends/narwhals/container.py
- **Verification:** `test_validate_polars_lazyframe` passes; all 125 narwhals tests pass
- **Committed in:** 9f9e38b (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - Bug)
**Impact on plan:** Essential correctness fix. The duck-typing approach in the plan was correct in intent but needed to operate on the class, not the instance. No scope creep.

## Issues Encountered

- Pre-existing ibis test failures (~88 failures) in tests/ibis/ — all pre-existing SQLite/ibis backend integration issues unrelated to ErrorHandler changes (documented in STATE.md)

## Next Phase Readiness

- NarwhalsErrorHandler is fully wired into all narwhals backends
- All narwhals tests pass (125 passed, 1 skipped, 3 xfailed, 4 xpassed)
- Plan 03 can proceed with remaining PR review items

---
*Phase: 01-pr-review-architecture-fixes*
*Completed: 2026-03-21*
