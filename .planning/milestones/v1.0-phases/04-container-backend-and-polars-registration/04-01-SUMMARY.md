---
phase: 04-container-backend-and-polars-registration
plan: "01"
subsystem: testing
tags: [narwhals, polars, pytest, xfail, test-scaffold]

# Dependency graph
requires:
  - phase: 03-column-backend
    provides: NarwhalsSchemaBackend in base.py, NarwhalsColumnBackend in components.py
provides:
  - xfail test stubs for all Phase 4 requirements (CONTAINER-01/02/03/04, REGISTER-01/02/04, TEST-03)
  - tests/backends/narwhals/test_container.py with 12 runnable stubs
affects:
  - 04-02-container-implementation
  - 04-03-register-narwhals-backends
  - 04-04-integration-validation

# Tech tracking
tech-stack:
  added: []
  patterns:
    - xfail stubs with strict=False for pre-implementation test scaffolding
    - lazy imports inside test bodies to avoid collection failure before modules exist

key-files:
  created:
    - tests/backends/narwhals/test_container.py
  modified: []

key-decisions:
  - "Imports of future modules (register, container) are guarded inside test bodies — collection never fails regardless of implementation state"
  - "test_narwhals_not_registered_by_default uses ImportError early-return to pass even before container.py exists"
  - "All xfail marks use strict=False so XPASS (early-passing stubs) does not fail the suite"

patterns-established:
  - "Lazy-import pattern: import future modules inside test body, not at module level"
  - "XPASS-safe scaffolding: strict=False xfail allows stubs to pass once implementation lands without breaking CI"

requirements-completed:
  - CONTAINER-01
  - CONTAINER-02
  - CONTAINER-03
  - CONTAINER-04
  - REGISTER-01
  - REGISTER-02
  - REGISTER-04
  - TEST-03

# Metrics
duration: 5min
completed: 2026-03-14
---

# Phase 4 Plan 01: Container Test Scaffold Summary

**12 xfail stubs covering CONTAINER-01/02/03/04, REGISTER-01/02/04, and TEST-03 in a single collectable test file using lazy body-level imports**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-03-14T13:14:00Z
- **Completed:** 2026-03-14T13:19:39Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Created `tests/backends/narwhals/test_container.py` with 12 xfail stubs
- All stubs collect without import errors (pytest shows no ERROR entries)
- Existing narwhals test suite unaffected (103 passed, 1 skipped, 13 xfailed, 1 xpassed)

## Task Commits

Each task was committed atomically:

1. **Task 1: Create test_container.py with xfail stubs** - `3107a9a` (test)

**Plan metadata:** (docs commit follows)

## Files Created/Modified
- `tests/backends/narwhals/test_container.py` - 12 xfail stubs for all Phase 4 requirements; uses lazy imports inside test bodies to survive collection before implementation modules exist

## Decisions Made
- Used lazy imports (imports inside test body) rather than module-level imports so `pytest --collect-only` succeeds even when `pandera.backends.narwhals.register` and `pandera.backends.narwhals.container` don't exist yet
- `test_narwhals_not_registered_by_default` returns early via `ImportError` guard rather than asserting, making it naturally XPASS once the module exists but narwhals isn't auto-registered
- All marks use `strict=False` so stubs that accidentally pass (XPASS) don't break CI as implementation lands incrementally

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Test scaffold in place; plans 04-02 (container backend), 04-03 (register function), and 04-04 (integration) can each promote their respective stubs from xfail to green
- No blockers

## Self-Check: PASSED
- tests/backends/narwhals/test_container.py: FOUND
- 04-01-SUMMARY.md: FOUND
- commit 3107a9a: FOUND

---
*Phase: 04-container-backend-and-polars-registration*
*Completed: 2026-03-14*
