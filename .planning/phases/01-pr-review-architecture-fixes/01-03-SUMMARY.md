---
phase: 01-pr-review-architecture-fixes
plan: 03
subsystem: validation-backend
tags: [narwhals, polars, lazy-frame, subsampling, capitalization]

# Dependency graph
requires:
  - phase: 01-02
    provides: NarwhalsErrorHandler wired into backends, polars coupling removed from container.py, capitalization partially fixed
provides:
  - validate() subsamples on nw.LazyFrame (no premature native materialization)
  - All 6 lowercase "narwhals" proper-noun instances capitalized to "Narwhals" in backend files
  - ROADMAP.md Plan 02 confirmed marked complete
affects: [01-pr-review-architecture-fixes]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Defer native materialization to final return: pass nw.LazyFrame through subsample(), normalize with .lazy() if nw.DataFrame, call _to_frame_kind_nw only at return"

key-files:
  created: []
  modified:
    - pandera/backends/narwhals/container.py
    - pandera/backends/narwhals/components.py
    - pandera/backends/narwhals/base.py

key-decisions:
  - "subsample() receives nw.LazyFrame directly; result normalized to LazyFrame via .lazy() if nw.DataFrame; _to_frame_kind_nw deferred to return statements only"
  - "drop_invalid_rows branch creates check_obj_parsed locally via _to_frame_kind_nw, returns immediately — no variable persists across the error-handler branch"

patterns-established:
  - "No native round-trips before checks: all validation logic operates on nw.LazyFrame until final materialization at return"

requirements-completed: [ARCH-02, ARCH-04]

# Metrics
duration: 2min
completed: 2026-03-22
---

# Phase 01 Plan 03: Gap Closure Summary

**validate() now defers native materialization until return, with subsampling on nw.LazyFrame and all 6 lowercase "narwhals" proper-noun instances capitalized to "Narwhals"**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-22T01:47:02Z
- **Completed:** 2026-03-22T01:48:50Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Fixed premature native materialization in validate(): subsample() now receives check_lf (nw.LazyFrame) directly, eliminating two unnecessary native round-trips
- Normalized subsample() result handling: if nw.DataFrame returned (head/tail path), wrapped with .lazy(); otherwise used as-is (nw.LazyFrame)
- Deferred all _to_frame_kind_nw calls to the final return statements and the drop_invalid_rows branch
- Fixed 6 lowercase "narwhals" proper-noun instances across container.py (4), components.py (1), base.py (1)
- Confirmed ROADMAP.md Plan 02 marker already set to [x] (documentation was already correct)

## Task Commits

Each task was committed atomically:

1. **Task 1: Fix validate() subsample ordering** - `42981c1` (fix)
2. **Task 2: Fix remaining capitalization nits and ROADMAP marker** - `c4cc0e8` (fix)

## Files Created/Modified

- `pandera/backends/narwhals/container.py` - Fixed subsample ordering in validate(); capitalized 4 docstring/comment instances
- `pandera/backends/narwhals/components.py` - Capitalized "Narwhals APIs" in class docstring
- `pandera/backends/narwhals/base.py` - Capitalized "Narwhals collect()" in inline comment

## Decisions Made

- subsample() receives nw.LazyFrame directly; result normalized to LazyFrame with .lazy() if nw.DataFrame — clean, no additional helper needed
- drop_invalid_rows branch creates check_obj_parsed locally (via _to_frame_kind_nw at entry to branch, returns immediately) to avoid any lingering reference to the variable outside the branch
- ROADMAP.md Plan 02 marker was already [x] — the VERIFICATION.md gap was a stale observation; no change needed

## Deviations from Plan

None - plan executed exactly as written. ROADMAP.md was already correct (already marked [x]) — this was a no-op discovery, not a deviation.

## Issues Encountered

None.

## Next Phase Readiness

- All three verification gaps from Plan 02 are now closed
- Phase 01 (PR Review Architecture Fixes) is complete: all 3 plans executed
- Requirements ARCH-01 through ARCH-04 fully satisfied
- Ready for next milestone work

---
*Phase: 01-pr-review-architecture-fixes*
*Completed: 2026-03-22*
