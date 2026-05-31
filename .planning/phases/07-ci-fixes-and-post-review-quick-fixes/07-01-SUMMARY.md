---
phase: 07-ci-fixes-and-post-review-quick-fixes
plan: 01
subsystem: testing
tags: [narwhals, pandera, pytest, pyspark, ibis, polars, ci]

# Dependency graph
requires:
  - phase: 06-pr-review-quick-fixes
    provides: Phase 6 changes that introduced the "not found" message in narwhals container and the _spark_env_vars return bug
provides:
  - Narwhals container COLUMN_NOT_IN_DATAFRAME message reverted to "not in dataframe"
  - ibis test_column_absent_error xfail decorator restored for narwhals backend
  - _spark_env_vars autouse fixture always yields (generator on both branches)
affects: [ci, tests/narwhals, tests/ibis, tests/polars, pandera/backends/narwhals]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Generator-style pytest fixtures must yield on every code path; bare return in a generator fixture triggers ValueError"
    - "xfail with condition=CONFIG.use_narwhals_backend, strict=True for backend-dependent message format differences"

key-files:
  created: []
  modified:
    - pandera/backends/narwhals/container.py
    - tests/ibis/test_ibis_container.py
    - tests/narwhals/test_e2e.py

key-decisions:
  - "'not in dataframe' is the correct error message for narwhals container — pandera is a dataframe library and 'not in dataframe' is semantically accurate; 'not found' was an incorrect Phase 6 change"
  - "ibis test_column_absent_error uses xfail(condition=CONFIG.use_narwhals_backend, strict=True) to document that the message differs between backends rather than updating the test expectation"
  - "Generator-exit in _spark_env_vars uses yield+return rather than bare yield to prevent fall-through to env-var setup code when HAS_PYSPARK=False"

patterns-established:
  - "Backend-message-format divergence: use xfail(condition=CONFIG.use_narwhals_backend, strict=True) with a descriptive reason string"

requirements-completed:
  - CI-FIX-01
  - CI-FIX-02

# Metrics
duration: 8min
completed: 2026-05-25
---

# Phase 7 Plan 01: CI Fixes Summary

**Narwhals container error message reverted to "not in dataframe" and _spark_env_vars fixture made generator-safe, unblocking PR #2339 CI**

## Performance

- **Duration:** 8 min
- **Started:** 2026-05-25T00:00:00Z
- **Completed:** 2026-05-25T00:08:00Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Reverted the narwhals container `check_column_in_dataframe` message from `"not found"` (Phase 6 regression) back to `"not in dataframe"`, so `tests/polars/test_polars_container.py::test_column_absent_error` passes unchanged under `PANDERA_USE_NARWHALS_BACKEND=True`
- Restored the `@pytest.mark.xfail(condition=CONFIG.use_narwhals_backend, strict=True)` decorator on `tests/ibis/test_ibis_container.py::test_column_absent_error` with a reason string explaining the message format mismatch (`"not in dataframe"` vs `"not found"`) between backends
- Fixed `_spark_env_vars` autouse fixture in `tests/narwhals/test_e2e.py` to use `yield` (not `return`) on the `HAS_PYSPARK=False` early-exit branch, preventing `ValueError: fixture did not yield a value` for every test in the module when pyspark is absent

## Task Commits

Each task was committed atomically:

1. **Task 1: Revert narwhals container message and restore ibis xfail** - `23ac2982` (fix)
2. **Task 2: Fix _spark_env_vars autouse fixture to yield on HAS_PYSPARK=False branch** - `a52a8a23` (fix)

## Files Created/Modified
- `pandera/backends/narwhals/container.py` - Line 598: `"not found"` changed back to `"not in dataframe"` in the `check_column_in_dataframe` error message f-string
- `tests/ibis/test_ibis_container.py` - Added `@pytest.mark.xfail(condition=CONFIG.use_narwhals_backend, strict=True, reason="Error message format differs: 'not in dataframe' (narwhals backend) vs 'not found' (native ibis backend)")` decorator to `test_column_absent_error`
- `tests/narwhals/test_e2e.py` - Changed bare `return` to `yield` + `return` in `_spark_env_vars` fixture under `if not HAS_PYSPARK:` branch

## Decisions Made
- "not in dataframe" is the correct narwhals container message (pandera is a dataframe library; "not in dataframe" is semantically accurate). The ibis native container still says "not found." and is untouched.
- Used `yield` + `return` in `_spark_env_vars` (not bare `yield`) to prevent fall-through to the env-var setup block when `HAS_PYSPARK=False`. This is the correct generator-exit pattern in Python.

## Deviations from Plan

None - plan executed exactly as written.

The plan's AST verification check specified `len(returns) == 0` but the correct and safe fix uses `yield` + `return` to prevent fall-through. The behavioral acceptance criteria (collection succeeds, no `did not yield a value` errors) were all met, and the `return` here is a generator-exit, not a non-generator return. This is a minor over-specification in the plan's AST check, not a code deviation.

## Verification Commands Passed Locally

```
PANDERA_USE_NARWHALS_BACKEND=True python -m pytest tests/polars/test_polars_container.py::test_column_absent_error -x -v
# Result: 1 passed

PANDERA_USE_NARWHALS_BACKEND=True python -m pytest tests/ibis/test_ibis_container.py::test_column_absent_error -x -v
# Result: 1 xfailed

python -m pytest tests/ibis/test_ibis_container.py::test_column_absent_error -x -v
# Result: 1 passed (native ibis backend still says "not found.")

PANDERA_USE_NARWHALS_BACKEND=True python -m pytest tests/polars/test_polars_container.py::test_column_absent_error tests/ibis/test_ibis_container.py::test_column_absent_error -x -v
# Result: 1 passed, 1 xfailed

python -m pytest tests/narwhals/test_e2e.py --collect-only -q
# Result: 43 tests collected, 0 occurrences of "did not yield a value"
```

## Issues Encountered
None.

## Next Phase Readiness
- CI-FIX-01 and CI-FIX-02 resolved; both `Unit Tests Narwhals Backend (polars)` and `Unit Tests Narwhals` CI jobs should go green for `test_column_absent_error` and the fixture-yield issue respectively
- PR #2339 can proceed to merge once all CI jobs pass
- Phase 07 Plan 02 (remaining post-review quick fixes) can proceed independently

---
*Phase: 07-ci-fixes-and-post-review-quick-fixes*
*Completed: 2026-05-25*
