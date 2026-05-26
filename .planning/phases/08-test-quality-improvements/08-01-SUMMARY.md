---
phase: 08-test-quality-improvements
plan: 01
subsystem: testing
tags: [pyspark, pytest, conftest, refactoring]

# Dependency graph
requires: []
provides:
  - "Module-level _cmp_errors helper in tests/pyspark/conftest.py (shared error-dict comparison)"
  - "test_pyspark_config.py TestPanderaConfig._cmp_errors delegates to conftest version"
  - "test_pyspark_error.py: 6 CONFIG ternaries removed; 3 DATA assertions use _cmp_errors"
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Shared error-dict comparison via conftest module-level function; explicit import alongside spark_df"
    - "DATA assertions drop 'error' key via _cmp_errors; SCHEMA assertions use direct equality"

key-files:
  created: []
  modified:
    - tests/pyspark/conftest.py
    - tests/pyspark/test_pyspark_config.py
    - tests/pyspark/test_pyspark_error.py

key-decisions:
  - "Module-level _cmp_errors in conftest.py requires explicit import (not pytest auto-injection) — plan assumed auto-injection which only works for fixtures"
  - "Inner helper named drop_error (no underscore prefix per naming conventions)"
  - "SCHEMA assertions left unchanged with direct equality; only DATA assertions with CONFIG ternaries use _cmp_errors"

patterns-established:
  - "tests/pyspark conftest non-fixture helpers: import explicitly alongside spark_df (from tests.pyspark.conftest import _cmp_errors, spark_df)"

requirements-completed: [TQ-01]

# Metrics
duration: 30min
completed: 2026-05-26
---

# Phase 08 Plan 01: _cmp_errors Extraction Summary

**Eliminated 6 `if CONFIG.use_narwhals_backend else` ternaries from PySpark test DATA assertions by extracting a shared `_cmp_errors` helper into `tests/pyspark/conftest.py` and rewiring both test files**

## Performance

- **Duration:** ~30 min
- **Started:** 2026-05-26T13:30:00Z
- **Completed:** 2026-05-26T14:01:15Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Added module-level `_cmp_errors(actual, expected)` to `tests/pyspark/conftest.py` — single source of truth for structural error-dict comparison that drops the `"error"` key before comparing fields
- Converted `TestPanderaConfig._cmp_errors` from a full implementation to a one-line delegation call
- Removed all 6 `if CONFIG.use_narwhals_backend else` ternaries from `test_pyspark_error.py` DATA assertions; also removed the now-unused `from pandera.config import CONFIG` import
- All 28 tests across both files pass under default and `PANDERA_USE_NARWHALS_BACKEND=True`

## Task Commits

Each task was committed atomically:

1. **Task 1: Extract _cmp_errors to conftest and convert TestPanderaConfig method to delegation** - `4fbeb406` (refactor)
2. **Task 2: Replace 6 CONFIG ternaries in test_pyspark_error.py with _cmp_errors on DATA assertions only** - `a96c0233` (refactor)

**Plan metadata:** (docs commit follows)

## Files Created/Modified
- `tests/pyspark/conftest.py` - Added module-level `_cmp_errors(actual, expected)` function after `config_params` fixture
- `tests/pyspark/test_pyspark_config.py` - Added `_cmp_errors` to explicit import; `TestPanderaConfig._cmp_errors` body replaced with single delegation call
- `tests/pyspark/test_pyspark_error.py` - Removed `from pandera.config import CONFIG`; added `from tests.pyspark.conftest import _cmp_errors, spark_df`; 6 ternaries removed; `"error"` keys dropped from 3 DATA expected dicts; 3 DATA assertions converted to `_cmp_errors(...)` calls

## Decisions Made
- Explicit import required (plan assumed pytest auto-injection, which only applies to fixtures not module-level functions)
- SCHEMA assertions left unchanged — they contain static backend-invariant error strings with no ternaries

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Added explicit import of `_cmp_errors` in both test files**
- **Found during:** Task 1 (extract _cmp_errors to conftest)
- **Issue:** Plan stated "pytest conftest auto-injection makes explicit import unnecessary." This is incorrect — pytest only auto-injects fixtures, not module-level functions. Calling `_cmp_errors(actual, expected)` by bare name in `TestPanderaConfig._cmp_errors` (a `@staticmethod`) raised `NameError: name '_cmp_errors' is not defined`.
- **Fix:** Added `_cmp_errors` to the existing `from tests.pyspark.conftest import ...` line in both `test_pyspark_config.py` and `test_pyspark_error.py`, matching the existing pattern used for `spark_df`.
- **Files modified:** `tests/pyspark/test_pyspark_config.py`, `tests/pyspark/test_pyspark_error.py`
- **Verification:** `pytest tests/pyspark/test_pyspark_config.py -x -q` exits 0; 16 tests pass
- **Committed in:** `4fbeb406` (Task 1 commit), `a96c0233` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 bug — incorrect auto-injection assumption in plan)
**Impact on plan:** Fix was minimal and consistent with the established import style (`spark_df` already imported the same way). No scope creep. The explicit import is arguably better style since it makes dependencies visible.

## Issues Encountered
None beyond the auto-fixed import deviation.

## Known Stubs
None.

## Threat Flags
None — no new trust boundaries, no production code changes, pure test refactoring.

## Self-Check
- [x] `tests/pyspark/conftest.py` contains `def _cmp_errors(actual, expected):` at module scope
- [x] `tests/pyspark/test_pyspark_config.py` `TestPanderaConfig._cmp_errors` has exactly one executable line
- [x] `grep -c 'if CONFIG.use_narwhals_backend' tests/pyspark/test_pyspark_error.py` = 0
- [x] `grep -c '_cmp_errors(' tests/pyspark/test_pyspark_error.py` = 3
- [x] Commits `4fbeb406` and `a96c0233` exist in git log
- [x] 28 tests pass under both backend modes

## Self-Check: PASSED

## Next Phase Readiness
- TQ-01 is closed — `test_pyspark_error.py` has zero backend-conditional ternaries in DATA assertions
- Remaining phase 08 plans (TQ-02, TQ-03, TQ-04) are independent of this plan's changes

---
*Phase: 08-test-quality-improvements*
*Completed: 2026-05-26*
