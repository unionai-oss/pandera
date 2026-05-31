---
phase: 02-remaining-pr-review-fixes
plan: 01
subsystem: testing
tags: [narwhals, polars, checks, backend]

# Dependency graph
requires:
  - phase: 01-pr-review-architecture-fixes
    provides: NarwhalsCheckBackend with IbisCheckBackend delegation, _materialize helper
provides:
  - postprocess_lazyframe_output using data_df.with_columns(results_df[CHECK_OUTPUT_KEY]) instead of horizontal concat
  - postprocess_bool_output using nw.get_native_namespace + nw.from_dict without polars-specific import
  - IbisCheckBackend delegation block with inline backward-compat documentation
affects: [02-remaining-pr-review-fixes]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Use DataFrame.with_columns(Series) instead of nw.concat(..., how='horizontal') for column attachment"
    - "Use nw.get_native_namespace(frame) + nw.from_dict(...).lazy() to create backend-agnostic LazyFrames"

key-files:
  created: []
  modified:
    - pandera/backends/narwhals/checks.py

key-decisions:
  - "data_df.with_columns(results_df[CHECK_OUTPUT_KEY]) is the correct pattern for attaching a single-column result to a data frame — avoids fragile positional alignment required by horizontal concat"
  - "nw.get_native_namespace accepts nw.LazyFrame and returns the underlying native module; nw.from_dict().lazy() creates a backend-agnostic lazy frame without importing polars directly"

patterns-established:
  - "Column attachment: prefer with_columns(series) over nw.concat horizontal for single-column appends"
  - "Backend-agnostic frame creation: nw.get_native_namespace + nw.from_dict instead of backend-specific constructors"

requirements-completed: []

# Metrics
duration: 2min
completed: 2026-03-22
---

# Phase 02 Plan 01: NarwhalsCheckBackend Refactor Summary

**Eliminated polars-specific coupling in postprocess_bool_output and fragile horizontal concat in postprocess_lazyframe_output using narwhals-native APIs**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-22T15:06:41Z
- **Completed:** 2026-03-22T15:08:30Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments
- Replaced `nw.concat([data_df, results_df], how="horizontal")` with `data_df.with_columns(results_df[CHECK_OUTPUT_KEY])` — eliminates positional alignment requirement and uses the semantically correct Narwhals API
- Replaced `import polars as pl; pl.LazyFrame(...)` with `nw.get_native_namespace(check_obj.frame)` + `nw.from_dict(...).lazy()` — postprocess_bool_output is now fully backend-agnostic
- All 125 narwhals backend tests pass at baseline (1 skipped, 3 xfailed, 4 xpassed)

## Task Commits

Each task was committed atomically:

1. **Task 1: Replace horizontal concat with with_columns in postprocess_lazyframe_output** - `edba528` (refactor)
2. **Task 2: Replace polars import with nw.get_native_namespace in postprocess_bool_output** - `1dad921` (refactor)

**Plan metadata:** (docs commit follows)

## Files Created/Modified
- `pandera/backends/narwhals/checks.py` - Updated postprocess_lazyframe_output and postprocess_bool_output methods

## Decisions Made
- `data_df.with_columns(results_df[CHECK_OUTPUT_KEY])` is the correct pattern — results_df[CHECK_OUTPUT_KEY] is a `nw.Series` which with_columns accepts directly, avoiding the alignment brittleness of horizontal concat
- `nw.get_native_namespace` accepts a nw.LazyFrame and returns the underlying native module (e.g., polars), enabling backend-agnostic eager frame creation via `nw.from_dict`; `.lazy()` wraps the result as a LazyFrame

## Deviations from Plan

None - plan executed exactly as written. The IbisCheckBackend delegation block already had inline documentation from prior work; no additional changes were needed there.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- checks.py refactor complete; ready for Plan 02 (custom checks delegation or check_dtype backend logic per ROADMAP)
- All 125 narwhals backend tests passing at baseline

---
*Phase: 02-remaining-pr-review-fixes*
*Completed: 2026-03-22*
