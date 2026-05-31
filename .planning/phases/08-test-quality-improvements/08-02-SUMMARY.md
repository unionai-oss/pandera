---
phase: 08-test-quality-improvements
plan: 02
subsystem: testing
tags: [narwhals, polars, failure-cases, pl-concat, tdd, regression-test]

# Dependency graph
requires:
  - phase: 07-ci-fixes-and-post-review-quick-fixes
    provides: stable narwhals backend with PySpark support
provides:
  - "_concat_failure_cases polars branch merges pl_items via pl.concat([lazy_result.collect()] + pl_items)"
  - "Regression test suite in tests/narwhals/test_concat_failure_cases.py asserting the merge behavior"
affects: [09, 10, future-narwhals-phases]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "TDD: failing test committed before production fix; confirmed RED then GREEN"
    - "pl.concat([lazy_result.collect()] + pl_items) pattern for mixing lazy+eager polars frames"

key-files:
  created:
    - tests/narwhals/test_concat_failure_cases.py
  modified:
    - pandera/backends/narwhals/base.py

key-decisions:
  - "Polars branch collects lazy result before pl.concat to avoid LazyFrame/DataFrame type mismatch (Pitfall #5)"
  - "No SchemaWarning in polars branch — polars can merge both sources cleanly unlike PySpark which lacks SparkSession"
  - "Updated docstring to accurately describe new polars branch behavior (lazy-only vs mixed cases)"

patterns-established:
  - "pl.concat([lazy.collect()] + pl_items): canonical pattern for mixing nw.LazyFrame results with eager pl.DataFrame items in the polars failure-case path"

requirements-completed: [TQ-02]

# Metrics
duration: 15min
completed: 2026-05-26
---

# Phase 08 Plan 02: _concat_failure_cases Polars Branch pl_items Merge Summary

**Silent pl_items drop in polars branch fixed via pl.concat([lazy_result.collect()] + pl_items), recovering schema-level failure rows previously lost from combined failure_cases frame**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-05-26T14:00:00Z
- **Completed:** 2026-05-26T14:05:00Z
- **Tasks:** 2 (TDD: RED + GREEN)
- **Files modified:** 2

## Accomplishments

- Fixed bug where polars branch of `_concat_failure_cases` silently dropped `pl_items` (native `pl.DataFrame` from `_build_eager_failure_case` / `_build_scalar_failure_case`) when `nw_items` were also present
- Added regression test suite with 3 tests: merge test (was RED, now GREEN), laziness-preservation test (was GREEN, stays GREEN), and no-warning test
- Updated docstring to accurately reflect that polars branch stays lazy only when no pl_items are present, and collects+merges when both types coexist

## Task Commits

Each task was committed atomically:

1. **Task 1: Write failing regression test for polars branch pl_items merging** - `b18750cc` (test)
2. **Task 2: Fix _concat_failure_cases polars branch to merge pl_items via pl.concat** - `4d820d66` (feat)

_Note: TDD plan — Task 1 committed in RED state (merge test failing); Task 2 committed in GREEN state (all 3 tests passing)_

## Files Created/Modified

- `tests/narwhals/test_concat_failure_cases.py` — Three regression tests targeting the polars branch: merge test, laziness-preservation test, and no-warning test
- `pandera/backends/narwhals/base.py` — Fixed polars branch (lines 109-122): `lazy_result = nw.to_native(nw.concat(nw_items))` + `if pl_items: return pl.concat([lazy_result.collect()] + pl_items)` + `return lazy_result`; updated docstring

## Decisions Made

- **Collect before concat:** `lazy_result.collect()` is required before passing to `pl.concat` — mixing `pl.LazyFrame` and `pl.DataFrame` in `pl.concat` raises a type error (Common Pitfall #5 from research)
- **No SchemaWarning:** Polars has no SparkSession barrier so both sources merge cleanly; the PySpark branch warns because it cannot convert `pl.DataFrame` to PySpark without a SparkSession — polars needs no such warning
- **Laziness preserved for all-lazy path:** When `pl_items` is empty, the function returns `lazy_result` (a `pl.LazyFrame`) unchanged, preserving the existing lazy behavior for pure data-check failures

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- TQ-02 closed: polars branch now correctly merges schema-level and data-check failure rows
- Regression tests provide permanent coverage against regressions in this behavior
- Phase 08 Plan 03 (behavioral test replacement for ARCH-03 source-inspection tests) can proceed independently

## Known Stubs

None.

## Threat Flags

T-08-02-T (Tampering) mitigated: fix STOPS silent data loss — pl_items were previously dropped, hiding schema-level failure rows from users. After fix, no failure rows are silently discarded on the polars path.

---

## Self-Check: PASSED

- [x] `tests/narwhals/test_concat_failure_cases.py` exists
- [x] `b18750cc` — test(08-02): add failing regression test for _concat_failure_cases polars pl_items merge (TQ-02)
- [x] `4d820d66` — feat(08-02): fix _concat_failure_cases polars branch to merge pl_items via pl.concat (TQ-02)
- [x] All 3 regression tests pass: `pytest tests/narwhals/test_concat_failure_cases.py` — 3 passed
- [x] No regressions in narwhals suite: 181 passed, 12 skipped, 4 xfailed

---
*Phase: 08-test-quality-improvements*
*Completed: 2026-05-26*
