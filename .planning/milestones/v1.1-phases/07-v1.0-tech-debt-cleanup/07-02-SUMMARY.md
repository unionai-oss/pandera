---
phase: 07-v1.0-tech-debt-cleanup
plan: "02"
subsystem: testing
tags: [narwhals, xfail, docstring, roadmap, hygiene]

# Dependency graph
requires:
  - phase: 07-v1.0-tech-debt-cleanup/07-01
    provides: _count_failure_cases dead branch removed, ibis API rename fixed
provides:
  - Accurate Check.native docstring describing nw.col(key) / nw.Expr protocol
  - 4 promoted tests (xfail markers removed, all pass)
  - test_drop_invalid_rows deleted (hollow test removed)
  - ROADMAP.md plan checkboxes accurate for all phases
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Check.native=False docstring: nw.col(key) (nw.Expr) as sole argument"

key-files:
  created: []
  modified:
    - pandera/api/checks.py
    - tests/backends/narwhals/test_container.py
    - tests/backends/narwhals/test_checks.py
    - .planning/ROADMAP.md

key-decisions:
  - "ROADMAP progress table restructured to reflect current 7-phase layout (v1.0 milestones moved to details block)"

patterns-established: []

requirements-completed: []

# Metrics
duration: 10min
completed: 2026-03-24
---

# Phase 7 Plan 02: Docs & Test Hygiene Summary

**Check.native=False docstring updated to nw.col(key)/nw.Expr protocol; 4 xfail markers promoted to strict passes; hollow test_drop_invalid_rows deleted; all ROADMAP checkboxes reconciled**

## Performance

- **Duration:** ~10 min
- **Started:** 2026-03-24T14:40:00Z
- **Completed:** 2026-03-24T14:50:00Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Updated `Check.native` docstring: `native=False` now correctly describes the Phase 5 protocol — check function receives `nw.col(key)` (a `nw.Expr`) as its sole argument
- Promoted 4 xfail tests to strict passing: `test_failure_cases_metadata`, `test_ibis_narwhals_auto_activated`, `test_ibis_backend_is_narwhals`, `test_postprocess_lazyframe_no_materialization_ibis`
- Deleted hollow `test_drop_invalid_rows` (used fake handler, `errors = []`, returned early — exercised no real logic)
- Marked all phase 02, 03, 05, 06, 07 plan checkboxes as `[x]` in ROADMAP.md; restructured progress table to show all 7 phases

## Task Commits

1. **Task 1: Update Check.native docstring and promote/delete xfail tests** - `f46d124` (fix)
2. **Task 2: Mark stale ROADMAP.md plan checkboxes as complete** - `f31dc13` (chore)

## Files Created/Modified
- `pandera/api/checks.py` - Updated native=False docstring to mention nw.col(key) and nw.Expr
- `tests/backends/narwhals/test_container.py` - Removed 3 xfail markers, deleted test_drop_invalid_rows
- `tests/backends/narwhals/test_checks.py` - Removed 1 xfail marker from test_postprocess_lazyframe_no_materialization_ibis
- `.planning/ROADMAP.md` - All plan checkboxes marked [x], progress table updated with all 7 phases

## Decisions Made
- ROADMAP progress table was restructured to properly list all 7 current phases (previous table only showed v1.0 phases 1-5 from old milestone, missing phases 6 and 7)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 7 is now complete (2/2 plans executed)
- Full narwhals test suite: 188 passed, 8 skipped, 4 xfailed, 1 pre-existing failure (`test_custom_check_ibis_lazy` — unrelated ibis API issue)
- v1.0 tech debt cleanup milestone complete

---
*Phase: 07-v1.0-tech-debt-cleanup*
*Completed: 2026-03-24*

## Self-Check: PASSED

- pandera/api/checks.py: FOUND
- tests/backends/narwhals/test_container.py: FOUND
- tests/backends/narwhals/test_checks.py: FOUND
- .planning/ROADMAP.md: FOUND
- 07-02-SUMMARY.md: FOUND
- Commit f46d124: FOUND
- Commit f31dc13: FOUND
