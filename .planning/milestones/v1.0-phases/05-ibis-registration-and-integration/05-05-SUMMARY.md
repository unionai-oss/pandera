---
phase: 05-ibis-registration-and-integration
plan: "05"
subsystem: backend
tags: [narwhals, ibis, pyarrow, failure_cases, polars, pl.concat]

requires:
  - phase: 05-04
    provides: ibis.Table failure_cases preserved lazily from run_check ibis path

provides:
  - failure_cases_metadata ibis/pyarrow branch that materializes ibis.Table and pyarrow.Table to pandas before pl.concat
affects:
  - tests/ibis/ test suite (closes Gap 2 from 05-VERIFICATION.md)
  - any future consumer of SchemaErrors.failure_cases after lazy ibis validation

tech-stack:
  added: []
  patterns:
    - "Dual ibis detection: ibis.Table (lazy expr) + pyarrow.Table (materialized) both flagged before scalar else branch"
    - "ibis.Table materialized via .execute(), pyarrow.Table via .to_pandas() — dispatch on hasattr(fc, 'execute')"

key-files:
  created: []
  modified:
    - pandera/backends/narwhals/base.py

key-decisions:
  - "pyarrow.lib.Table also detected as ibis-originated failure_cases — narwhals collect() on ibis-backed LazyFrame materializes to pyarrow, not pandas"
  - "ibis.Table materialization uses .execute() (returns pandas); pyarrow.Table uses .to_pandas() — both dispatch on hasattr(_ibis_fc, 'execute')"
  - "ibis check_output index computation: execute check_output ibis.Table and extract row indices where CHECK_OUTPUT_KEY is False"

patterns-established:
  - "ibis-originated failure_cases detection: check for both ibis.Table and pyarrow.Table before the scalar else branch"

requirements-completed: [REGISTER-03, TEST-04]

duration: 4min
completed: 2026-03-15
---

# Phase 5 Plan 5: Ibis failure_cases_metadata Gap Closure Summary

**ibis.Table and pyarrow.Table failure_cases materialized to pandas in failure_cases_metadata, closing Gap 2 (pl.concat Object dtype crash) so lazy ibis validation surfaces multiple errors without crashing**

## Performance

- **Duration:** ~4 min
- **Started:** 2026-03-15T06:04:18Z
- **Completed:** 2026-03-15T06:09:12Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Closed Gap 2 from VERIFICATION.md: `pl.concat` in `failure_cases_metadata` no longer crashes with `SchemaError: type Object incompatible with expected type String` when ibis-originated failure cases are in the error collection
- Added ibis.Table detection (lazy ibis expr — from ibis run_check path) and pyarrow.Table detection (materialized — from narwhals collect on ibis-backed frame) before the scalar `else` branch
- `test_lazy_validation_errors` in `tests/ibis/test_ibis_container.py` now passes: 6 failure cases correctly collected and surfaced via SchemaErrors
- Narwhals backend suite remains fully green (124 passed)

## Task Commits

Each task was committed atomically:

1. **Task 1: Add ibis.Table branch to failure_cases_metadata** - `a022bed` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified
- `pandera/backends/narwhals/base.py` - Added ibis-originated failure_cases detection block (ibis.Table + pyarrow.Table) and elif branch that materializes to pandas, converts to pl.DataFrame, computes index from ibis check_output, and casts to same schema as pl.DataFrame branch

## Decisions Made
- pyarrow.lib.Table must also be detected — narwhals `.collect()` on an ibis-backed LazyFrame returns pyarrow, not an ibis lazy expression. Without this, the dtype-error failure_cases (which go through narwhals collection) would still fall through to the scalar else branch.
- Dispatch between ibis.Table (`.execute()`) and pyarrow.Table (`.to_pandas()`) uses `hasattr(_ibis_fc, 'execute')` — avoids importing pyarrow/ibis twice in the materialization step.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Extended ibis detection to also catch pyarrow.lib.Table**
- **Found during:** Task 1 (verification — test_lazy_validation_errors still crashed)
- **Issue:** Plan specified detecting `ibis.Table` failure_cases, but the actual failure_cases from narwhals-path ibis validation are `pyarrow.lib.Table` (narwhals `.collect()` on ibis-backed LazyFrame materializes to pyarrow). `isinstance(fc, ibis.Table)` returned `False` for pyarrow.Table, so ibis branch was never entered.
- **Fix:** Added a second detection block after the ibis.Table check: `import pyarrow; if isinstance(err.failure_cases, pa.Table): _ibis_fc = err.failure_cases`. Then in the elif branch, dispatch on `hasattr(_ibis_fc, 'execute')` to choose `.execute()` (ibis.Table) vs `.to_pandas()` (pyarrow.Table).
- **Files modified:** pandera/backends/narwhals/base.py
- **Verification:** `test_lazy_validation_errors` passes; narwhals suite passes 124 tests; ibis container test count improved from 31 failures to 30 failures
- **Committed in:** `a022bed` (part of Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Fix required for correctness — pyarrow is the actual runtime type of ibis-originated failure_cases through the narwhals path. No scope creep.

## Issues Encountered
- Initial debug was complicated because a direct Python script used `pa.Column` (pandas) instead of `pandera.ibis.Column` (ibis-api), causing a `BackendNotFoundError` that masked the actual `failure_cases_metadata` crash. Narrowing to the correct test via pytest traceback revealed the true crash site.

## Next Phase Readiness
- All three gaps from VERIFICATION.md are now closed (Gap 1 and Gap 3 by plan 05-04, Gap 2 by this plan)
- `test_lazy_validation_errors` passes — ibis lazy=True validation can collect and surface multiple errors
- Remaining ibis test failures (30 pre-existing) are unrelated to failure_cases_metadata: drop_invalid_rows, regex_selector, element_wise UDF limitations, and n_failure_cases counting

---
*Phase: 05-ibis-registration-and-integration*
*Completed: 2026-03-15*
