---
phase: 08-fix-lazy-true-critical-regressions
plan: "01"
subsystem: testing
tags: [pytest, narwhals, polars, ibis, lazy-validation, regression-tests]

# Dependency graph
requires:
  - phase: 07
    provides: "nw.from_native(failure_cases, eager_only=False) unified pattern in _count_failure_cases"
  - phase: 06-eliminate-unnecessary-materialization-lazy-first-failure-cases-and-check-output
    provides: "Phase 6 contract: failure_cases is always native at SchemaError boundary"
provides:
  - "RED baseline regression tests for MISSING-01 (polars lazy failure_cases repr collapse) and MISSING-02 (bool scalar TypeError in _count_failure_cases)"
  - "tests/backends/narwhals/test_lazy_regressions.py with 3 test functions covering both bugs"
affects: [08-02-fix-production-code]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "native=True check with (native_frame, key) -> bool signature triggers the failure_cases=False path in run_check"
    - "element_wise=True with lambda returning False does NOT trigger MISSING-02 — polars raises TypeError from map_batches before run_check completes"

key-files:
  created:
    - tests/backends/narwhals/test_lazy_regressions.py
  modified: []

key-decisions:
  - "ibis MISSING-01 test is GREEN (not RED) because Phase 6 already fixed ibis.Table rewrapping — test retained as regression guard"
  - "MISSING-02 test uses native=True bool-returning check (lambda native_frame, key: False) to correctly trigger failure_cases=False path — element_wise=True approach fails at polars map_batches level before reaching the bug"
  - "plan's element_wise=True lambda for MISSING-02 does not exercise the bug — the TypeError is caught by the outer except Exception in run_checks and converted to string failure_case before reaching _count_failure_cases"

patterns-established:
  - "MISSING-02 trigger pattern: Check(lambda native_frame, key: False) with two arguments is the correct trigger for the bool scalar path"

requirements-completed: [MISSING-01, MISSING-02]

# Metrics
duration: 61min
completed: 2026-03-25
---

# Phase 8 Plan 01: Fix lazy=True Critical Regressions RED Baseline Summary

**Regression test file for MISSING-01 (polars lazy failure_cases repr collapse) and MISSING-02 (bool scalar TypeError crash), with two of three tests correctly RED on unmodified codebase**

## Performance

- **Duration:** 61 min
- **Started:** 2026-03-25T00:57:40Z
- **Completed:** 2026-03-25T02:00:00Z
- **Tasks:** 1
- **Files modified:** 1 (created)

## Accomplishments
- Created `tests/backends/narwhals/test_lazy_regressions.py` with 3 test functions
- Confirmed MISSING-01 polars test is RED: `len(fc) == 1` (repr string of pl.DataFrame) instead of 3
- Confirmed MISSING-02 test is RED: `TypeError: Unsupported dataframe type, got: <class 'bool'>` from `_count_failure_cases`
- Discovered ibis MISSING-01 path already fixed by Phase 6 — ibis test documents correct behavior as regression guard

## Task Commits

Each task was committed atomically:

1. **Task 1: Write RED regression tests for MISSING-01 (polars + ibis) and MISSING-02** - `2cee1ae` (test)

## Files Created/Modified
- `tests/backends/narwhals/test_lazy_regressions.py` - Three regression tests: polars lazy failure_cases row count, ibis lazy failure_cases Table type/count, bool scalar check TypeError crash

## Decisions Made

- ibis MISSING-01 test is GREEN because Phase 6 already fixed the ibis.Table rewrap path in `failure_cases_metadata()`. Test retained as regression guard — it documents the correct expected behavior.
- MISSING-02 test required native=True check signature `(native_frame, key) -> bool` to correctly trigger the bug. The plan's suggested `element_wise=True, lambda x: False` triggers a different TypeError from polars `map_batches` that is caught by the outer `except Exception` handler in `run_checks` and converted to a string failure_case before reaching `_count_failure_cases`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] MISSING-02 test corrected to use native=True bool-returning check**
- **Found during:** Task 1 (Write RED regression tests)
- **Issue:** Plan specified `Check(lambda x: False, element_wise=True)` for MISSING-02. This does not trigger `_count_failure_cases(False)` — polars raises TypeError from `map_batches` ("`map` with `returns_scalar=False` must return a Series") which is caught by the outer `except Exception` handler and converted to a string failure_case, bypassing the bug.
- **Fix:** Changed to `Check(lambda native_frame, key: False)` — a two-argument lambda triggers the `native=True` dispatch path, which calls `check_fn(native_frame, key)` returning `bool False`, then `postprocess_bool_output` sets `failure_cases=None`, and `run_check` sets `failure_cases = passed = False`. This causes `_count_failure_cases(False)` to be called, triggering the actual TypeError.
- **Files modified:** tests/backends/narwhals/test_lazy_regressions.py
- **Verification:** `pytest tests/backends/narwhals/test_lazy_regressions.py::test_lazy_bool_output_check_does_not_crash -v` shows FAILED with `TypeError: Unsupported dataframe type, got: <class 'bool'>`
- **Committed in:** 2cee1ae (Task 1 commit)

**2. [Rule 1 - Bug] ibis MISSING-01 test uses IbisSchema + IbisColumn (not DataFrameSchema + polars Column)**
- **Found during:** Task 1 (Write RED regression tests)
- **Issue:** Plan specified using `pandera.api.polars.container.DataFrameSchema` with ibis.Table input. This raises `BackendNotFoundError` (not a MISSING-01 assertion failure) because polars DataFrameSchema doesn't handle ibis.Table.
- **Fix:** Changed to `pandera.api.ibis.container.DataFrameSchema` with `IbisColumn(dt.int64)` for the ibis test. This correctly exercises the ibis lazy=True validation path.
- **Files modified:** tests/backends/narwhals/test_lazy_regressions.py
- **Verification:** Test passes because ibis MISSING-01 was already fixed by Phase 6.
- **Committed in:** 2cee1ae (Task 1 commit)

---

**Total deviations:** 2 auto-fixed (both Rule 1 - Bug: incorrect test trigger patterns)
**Impact on plan:** Both fixes necessary for tests to correctly exercise the actual bugs rather than unrelated failure modes. No scope creep.

## Issues Encountered
- ibis MISSING-01 path is already GREEN because Phase 6 rewrapped ibis.Table in `failure_cases_metadata()` via the unified `try: nw.from_native / except TypeError` pattern. The research doc describes the pre-fix state; Phase 6 already addressed the ibis side of MISSING-01.
- Two of three tests are RED on the current codebase: polars MISSING-01 (assertion failure) and MISSING-02 (TypeError). The ibis MISSING-01 test is GREEN and serves as a regression guard.

## Next Phase Readiness
- RED baseline established for polars MISSING-01 and MISSING-02
- ibis MISSING-01 already GREEN (Phase 6 fix confirmed working)
- Plan 08-02 can apply the two surgical fixes: unified `nw.from_native` try/except in `failure_cases_metadata()` and `try/except TypeError` restoration in `_count_failure_cases()`
- After Plan 08-02 fixes: all three tests should be GREEN

---
*Phase: 08-fix-lazy-true-critical-regressions*
*Completed: 2026-03-25*
