---
phase: 02-check-backend
plan: 01
subsystem: testing
tags: [narwhals, pytest, xfail, conftest, check-backend]

# Dependency graph
requires:
  - phase: 01-foundation
    provides: "NarwhalsData(frame, key) NamedTuple and NarwhalsCheckResult types"
provides:
  - "pandera.backends.narwhals package (empty __init__.py)"
  - "tests/backends/narwhals/conftest.py with make_narwhals_frame fixture parameterized over polars and ibis"
  - "tests/backends/narwhals/test_checks.py with 5 stub test functions (62 total test cases) all xfail(strict=True)"
affects: [02-02, 02-03, phase-03, phase-04, phase-05]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "pytest fixture parameterized over polars/ibis backends via params=[polars, ibis]"
    - "autouse=True scope=module fixture with fully guarded imports for pre-existing files"
    - "xfail(strict=True) stubs that flip to passing once backend implementation lands"

key-files:
  created:
    - pandera/backends/narwhals/__init__.py
    - tests/backends/narwhals/conftest.py
    - tests/backends/narwhals/test_checks.py
  modified: []

key-decisions:
  - "Both NarwhalsCheckBackend and builtin_checks imports in conftest fixture are guarded with try/except ImportError so autouse fixture does not break dtype tests before checks.py exists"
  - "test_element_wise_sql_lazy_raises uses pytest.skip for polars parameterization (not ibis-only fixture) to avoid xfail with never-failing polars path"

patterns-established:
  - "Pattern 1: All conftest imports of not-yet-existing backend files guarded with try/except ImportError"
  - "Pattern 2: Builtin check test cases parametrized as BUILTIN_CHECK_CASES list of pytest.param tuples shared across pass/fail tests"

requirements-completed: [TEST-01]

# Metrics
duration: 8min
completed: 2026-03-09
---

# Phase 2 Plan 1: Narwhals Check Backend Test Scaffold Summary

**pytest fixture scaffold for narwhals check backend: make_narwhals_frame parameterized over polars/ibis, 62 xfail stubs covering CHECKS-01/02/03, and package init unblocking Plans 02-02 and 02-03**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-10T01:52:56Z
- **Completed:** 2026-03-10T02:01:00Z
- **Tasks:** 1
- **Files modified:** 3

## Accomplishments
- Created `pandera/backends/narwhals/__init__.py` making narwhals a proper Python package
- Created `tests/backends/narwhals/conftest.py` with `make_narwhals_frame` fixture parameterized over polars and ibis, plus module-scoped autouse registration fixture with fully guarded imports
- Created `tests/backends/narwhals/test_checks.py` with 5 stub test functions (62 total test cases from parametrization) all marked `xfail(strict=True)` covering CHECKS-01, CHECKS-02, and CHECKS-03

## Task Commits

Each task was committed atomically:

1. **Task 1: Create package init and test scaffold** - `2494c93` (feat)

**Plan metadata:** (docs commit to follow)

## Files Created/Modified
- `pandera/backends/narwhals/__init__.py` - Empty package init with module docstring
- `tests/backends/narwhals/conftest.py` - make_narwhals_frame fixture (polars + ibis) and autouse backend registration fixture with guarded imports
- `tests/backends/narwhals/test_checks.py` - 5 stub test functions, 14-check parametrize list (BUILTIN_CHECK_CASES), all xfail(strict=True)

## Decisions Made
- Guarded `NarwhalsCheckBackend` import in `_register_narwhals_check_backend` fixture (identical to builtin_checks guard) so `autouse=True, scope="module"` doesn't break dtype tests before `checks.py` exists
- `test_element_wise_sql_lazy_raises` uses `pytest.skip` for the polars parameterization variant to avoid an xfail test that would never fail (polars supports `map_batches`, only ibis raises `NotImplementedError`)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Guarded NarwhalsCheckBackend import in autouse fixture**
- **Found during:** Task 1 (Create package init and test scaffold)
- **Issue:** `_register_narwhals_check_backend` fixture is `autouse=True, scope="module"` so it runs for all tests in `tests/backends/narwhals/` including `test_narwhals_dtypes.py`. The unguarded `from pandera.backends.narwhals.checks import NarwhalsCheckBackend` import caused `ModuleNotFoundError` during fixture setup, turning all 26 dtype tests from PASSED to ERROR.
- **Fix:** Wrapped the `NarwhalsCheckBackend` import and `Check.register_backend(...)` call in a `try/except ImportError` block (same pattern as the existing `builtin_checks` guard below it)
- **Files modified:** `tests/backends/narwhals/conftest.py`
- **Verification:** `pytest tests/backends/narwhals/ -q` shows 26 passed, 1 skipped, 61 xfailed — no errors
- **Committed in:** `2494c93` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Fix required for correctness — existing dtype tests would error without it. No scope creep.

## Issues Encountered
None beyond the auto-fixed fixture guard issue above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Test scaffold is in place; Plans 02-02 and 02-03 can now target these stubs
- Plan 02-02 (`NarwhalsCheckBackend` + `checks.py`) will flip `test_builtin_check_routing`, `test_user_defined_check_routing`, `test_builtin_checks_pass`, `test_builtin_checks_fail` from xfail to passing
- Plan 02-03 (`builtin_checks.py`) completes the full check dispatch loop
- `test_element_wise_sql_lazy_raises[ibis]` will pass once checks.py handles `map_batches` NotImplementedError

---
*Phase: 02-check-backend*
*Completed: 2026-03-09*
