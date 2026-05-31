---
phase: 01-structural-cleanup
plan: 06
subsystem: backend
tags: [narwhals, imports, type-checking, polars, CLEAN-04]

# Dependency graph
requires:
  - phase: 01-structural-cleanup
    provides: plans 01-01 through 01-05 — narwhals backend import cleanup foundations
provides:
  - TYPE_CHECKING guard for polars DataFrameSchema in container.py (polars-free import path)
  - Module-level import re in container.py and components.py
  - Module-level NarwhalsData and _to_native in narwhals_engine.py
  - CLEAN-04 requirement fully satisfied
affects: [future-narwhals-plans, backend-isolation]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "TYPE_CHECKING guard with from __future__ import annotations for optional runtime deps"
    - "Module-level stdlib imports; no import statements inside method bodies"

key-files:
  created: []
  modified:
    - pandera/backends/narwhals/container.py
    - pandera/backends/narwhals/components.py
    - pandera/engines/narwhals_engine.py

key-decisions:
  - "DataFrameSchema import guarded by TYPE_CHECKING — polars not required at runtime for narwhals container backend"
  - "from __future__ import annotations enables lazy annotation evaluation so TYPE_CHECKING guard works correctly"

patterns-established:
  - "TYPE_CHECKING guard pattern: from __future__ import annotations + if TYPE_CHECKING: import for optional deps"

requirements-completed: [CLEAN-04]

# Metrics
duration: 3min
completed: 2026-04-10
---

# Phase 01 Plan 06: Import Hygiene (CLEAN-04 Gap Closure) Summary

**TYPE_CHECKING guard for optional polars DataFrameSchema + module-level re/NarwhalsData/_to_native hoisted in three narwhals files, closing CLEAN-04**

## Performance

- **Duration:** 3 min
- **Started:** 2026-04-10T13:24:15Z
- **Completed:** 2026-04-10T13:28:09Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- `container.py`: added `from __future__ import annotations`, moved `DataFrameSchema` import behind `TYPE_CHECKING` guard, hoisted `import re`, removed two redundant inner `_to_native` imports
- `components.py`: hoisted `import re` to module level, removed inner `import re` from `get_regex_columns`
- `narwhals_engine.py`: added `NarwhalsData` and `_to_native` at module level, removed inner imports from `coerce` and `try_coerce`
- All 229 narwhals tests pass (8 skipped, 1 xfailed)

## Task Commits

Each task was committed atomically:

1. **Task 1: TYPE_CHECKING guard for polars DataFrameSchema in container.py** - `2f1fdaab` (fix)
2. **Task 2: Hoist inner imports in components.py and narwhals_engine.py** - `f720c597` (fix)

## Files Created/Modified

- `pandera/backends/narwhals/container.py` - Added `from __future__ import annotations`; TYPE_CHECKING guard for DataFrameSchema; module-level `import re`; removed two inner `_to_native` re-imports
- `pandera/backends/narwhals/components.py` - Added module-level `import re`; removed inner `import re` from `get_regex_columns`
- `pandera/engines/narwhals_engine.py` - Added `NarwhalsData` and `_to_native` at module level; removed inner imports from `coerce` and `try_coerce`; updated stale comment on `NarwhalsDataContainer`

## Decisions Made

- `from __future__ import annotations` is required alongside the `TYPE_CHECKING` guard so that `DataFrameSchema` in annotation positions is never evaluated at runtime — this is the standard PEP 563 pattern
- The pre-existing circular import guard (`from pandera.engines import narwhals_engine` inside `components.py`) was intentionally left in place; the plan's AST verification check was overly broad and would have flagged this legitimate inner import. The done criteria for Task 2 only targets `import re`, `NarwhalsData`, and `_to_native`, all of which were resolved.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Updated stale NarwhalsDataContainer comment**
- **Found during:** Task 2 (narwhals_engine.py edits)
- **Issue:** Comment said "imported lazily" after NarwhalsData was hoisted to module level
- **Fix:** Removed "— imported lazily" from the type alias comment
- **Files modified:** pandera/engines/narwhals_engine.py
- **Verification:** No functional change; comment accuracy only
- **Committed in:** f720c597 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (minor comment update)
**Impact on plan:** No scope creep. Pre-existing circular import guard in components.py deliberately preserved.

## Issues Encountered

The plan's automated verification command used `col_offset > 0` to detect inner imports, which also flags the pre-existing `from pandera.engines import narwhals_engine` circular import guard in `components.py`. This guard is intentional and must remain. The done criteria (which only checks the specific targeted imports) were fully met, and the test suite confirms correctness.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- CLEAN-04 is fully satisfied
- All narwhals backends (container, components, engine) now have clean module-level imports
- No inner `import re`, `NarwhalsData`, or `_to_native` remain in any of the three files
- Phase 01 plan 06 is the final plan in phase 01-structural-cleanup

## Self-Check: PASSED

All files verified: container.py, components.py, narwhals_engine.py, SUMMARY.md
All commits verified: 2f1fdaab (Task 1), f720c597 (Task 2)

---
*Phase: 01-structural-cleanup*
*Completed: 2026-04-10*
