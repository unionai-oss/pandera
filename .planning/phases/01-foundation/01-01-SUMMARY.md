---
phase: 01-foundation
plan: 01
subsystem: infra
tags: [narwhals, narwhals.stable.v1, NamedTuple, pytest, xfail]

# Dependency graph
requires: []
provides:
  - narwhals optional-dependency extra in pyproject.toml (narwhals >= 2.15.0)
  - pandera.api.narwhals package with NarwhalsData and NarwhalsCheckResult types
  - _to_native helper with pass_through=True for safe native frame unwrapping
  - tests/backends/narwhals/ test scaffold with INFRA tests GREEN and ENGINE stubs xfail
affects:
  - 01-02 (narwhals engine — imports NarwhalsData, NarwhalsCheckResult, _to_native)
  - 01-03 (schema model — uses NarwhalsData as container type)
  - 01-04 (engine registration — opt-in path, must not side-effect plain imports)

# Tech tracking
tech-stack:
  added: [narwhals >= 2.15.0]
  patterns:
    - Use narwhals.stable.v1 (not bare narwhals) for API stability
    - NarwhalsData uses field name 'frame' (not 'lazyframe' like Polars PolarsData)
    - _to_native uses pass_through=True so already-native frames are safe to pass
    - xfail stubs establish test RED state before Plan 02 implementation

key-files:
  created:
    - pyproject.toml (modified — narwhals extra added)
    - pandera/api/narwhals/__init__.py
    - pandera/api/narwhals/types.py
    - pandera/api/narwhals/utils.py
    - tests/backends/__init__.py
    - tests/backends/narwhals/__init__.py
    - tests/backends/narwhals/test_narwhals_dtypes.py
  modified:
    - pyproject.toml

key-decisions:
  - "Use narwhals.stable.v1 (not bare narwhals) for API stability across narwhals versions"
  - "NarwhalsData.frame field named 'frame' (not 'lazyframe') to distinguish from Polars PolarsData"
  - "_to_native uses pass_through=True to safely handle both narwhals-wrapped and already-native frames"
  - "ENGINE tests marked xfail to maintain RED state until Plan 02 implements narwhals_engine.py"

patterns-established:
  - "Narwhals imports: always import narwhals.stable.v1 as nw"
  - "Named tuple pattern: NarwhalsData mirrors PolarsData but uses 'frame' not 'lazyframe'"
  - "Test scaffold: xfail stubs for unimplemented plans, passing tests for implemented code"

requirements-completed: [INFRA-01, INFRA-02, INFRA-03]

# Metrics
duration: 2min
completed: 2026-03-09
---

# Phase 1 Plan 01: Narwhals API Package and Test Scaffold Summary

**narwhals.stable.v1 API package with NarwhalsData/NarwhalsCheckResult types, _to_native utility, and pytest test scaffold with 4 INFRA tests GREEN and 22 ENGINE stubs xfail**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-09T22:47:48Z
- **Completed:** 2026-03-09T22:49:48Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments

- Added `narwhals = ["narwhals >= 2.15.0"]` to `[project.optional-dependencies]` in pyproject.toml
- Created `pandera/api/narwhals/` package with `NarwhalsData` (frame/key fields) and `NarwhalsCheckResult` (4 LazyFrame fields) named tuples
- Created `_to_native` helper that safely handles both narwhals-wrapped and already-native frames via `pass_through=True`
- Created `tests/backends/narwhals/` test scaffold: 4 INFRA tests passing GREEN, 22 ENGINE xfail stubs waiting for Plan 02

## Task Commits

Each task was committed atomically:

1. **Task 1: Add narwhals extra to pyproject.toml and create API package** - `7df9650` (feat)
2. **Task 2: Create test scaffold for Phase 1** - `5841155` (test)

**Plan metadata:** (docs commit to follow)

_Note: TDD tasks have implementation and tests in separate task commits_

## Files Created/Modified

- `pyproject.toml` - Added narwhals >= 2.15.0 optional-dependency extra
- `pandera/api/narwhals/__init__.py` - Package init (minimal docstring)
- `pandera/api/narwhals/types.py` - NarwhalsData and NarwhalsCheckResult NamedTuples using narwhals.stable.v1
- `pandera/api/narwhals/utils.py` - _to_native helper with pass_through=True
- `tests/backends/__init__.py` - Empty package init for backends test directory
- `tests/backends/narwhals/__init__.py` - Empty package init for narwhals test sub-package
- `tests/backends/narwhals/test_narwhals_dtypes.py` - INFRA-02/03 tests (GREEN) and ENGINE-01/02/03 stubs (xfail)

## Decisions Made

- Used `narwhals.stable.v1` (not bare `narwhals`) consistent with project decision to insulate from breaking API changes
- Named the frame field `frame` (not `lazyframe`) in `NarwhalsData` as specified — distinguishes from Polars `PolarsData.lazyframe`
- `_to_native` uses `pass_through=True` so callers don't need to track whether a frame is already native
- ENGINE tests marked `@pytest.mark.xfail` with explicit reason string pointing to Plan 02

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `NarwhalsData`, `NarwhalsCheckResult`, and `_to_native` are importable and ready for Plan 02 to use
- Test scaffold is in place — Plan 02 only needs to implement `narwhals_engine.py` and the xfail tests will flip to GREEN
- No side effects: importing `pandera.api.narwhals` does not register any engine dtypes

---
*Phase: 01-foundation*
*Completed: 2026-03-09*

## Self-Check: PASSED

All files found, all commits verified.
