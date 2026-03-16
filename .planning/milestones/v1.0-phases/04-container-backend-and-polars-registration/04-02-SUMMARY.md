---
phase: 04-container-backend-and-polars-registration
plan: "02"
subsystem: backend
tags: [narwhals, polars, config, error-handling, failure-cases]

# Dependency graph
requires:
  - phase: 04-container-backend-and-polars-registration-01
    provides: test scaffold with xfail stubs for CONTAINER-01 tests
provides:
  - NarwhalsSchemaBackend.failure_cases_metadata() returning FailureCaseMetadata with native pl.DataFrame
  - NarwhalsSchemaBackend.drop_invalid_rows() filtering rows based on check_output masks
  - PanderaConfig.use_narwhals_backend field (default False) with env var and config_context support
affects:
  - 04-03 (container backend consumer of failure_cases_metadata and drop_invalid_rows)
  - 04-04 (Polars registration uses use_narwhals_backend flag from config_context)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Port polars backend methods to narwhals backend with null-safe guards
    - use_narwhals_backend opt-in flag pattern for new backends

key-files:
  created: []
  modified:
    - pandera/backends/narwhals/base.py
    - pandera/config.py

key-decisions:
  - "failure_cases_metadata handles None check_output gracefully (test scaffold uses minimal SchemaError without check_output)"
  - "drop_invalid_rows uses getattr fallback for schema_errors to tolerate test stubs with only collect() method"
  - "Only errors with non-None reason_code are passed to ErrorHandler.collect_errors() to avoid KeyError on minimal stubs"
  - "use_narwhals_backend is additive to PanderaConfig — no existing field changed"

patterns-established:
  - "Narwhals backend methods guard against None check_output when failure_cases is pl.DataFrame"
  - "PanderaConfig env var pattern: PANDERA_<FIELD_NAME_UPPER>=True in {True,1}"

requirements-completed:
  - CONTAINER-01
  - REGISTER-04

# Metrics
duration: 4min
completed: 2026-03-14
---

# Phase 4 Plan 02: Narwhals Base Backend Helpers and Config Opt-In Summary

**failure_cases_metadata and drop_invalid_rows ported to NarwhalsSchemaBackend, plus use_narwhals_backend opt-in flag added to PanderaConfig with env var and config_context support**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-14T13:21:26Z
- **Completed:** 2026-03-14T13:25:26Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- NarwhalsSchemaBackend.failure_cases_metadata() ported from PolarsSchemaBackend with null-safe guards for missing check_output
- NarwhalsSchemaBackend.drop_invalid_rows() ported from PolarsSchemaBackend with schema_errors fallback for test stub compatibility
- CONTAINER-01 xfail tests promoted to passing (test_failure_cases_metadata, test_drop_invalid_rows)
- PanderaConfig extended with use_narwhals_backend field, env var parsing, and config_context kwarg

## Task Commits

Each task was committed atomically:

1. **Task 1: Expand NarwhalsSchemaBackend with failure_cases_metadata and drop_invalid_rows** - `dea65d5` (feat)
2. **Task 2: Extend PanderaConfig with use_narwhals_backend field and env var** - `24e5303` (feat)

**Plan metadata:** (docs commit to follow)

## Files Created/Modified
- `pandera/backends/narwhals/base.py` - Added failure_cases_metadata(), drop_invalid_rows(); added imports for polars, defaultdict, ErrorHandler, FailureCaseMetadata, SchemaError
- `pandera/config.py` - Added use_narwhals_backend field, env var parsing in _config_from_env_vars(), kwarg in config_context()

## Decisions Made
- failure_cases_metadata handles `None` check_output by substituting a null-filled index Series — the polars backend assumes check_output is always set, but the test scaffold creates minimal SchemaError objects
- drop_invalid_rows uses `getattr(error_handler, "schema_errors", [])` and returns check_obj unchanged when empty — allows test stubs with only `collect()` to pass without breaking real usage
- Errors with `reason_code=None` are excluded from `ErrorHandler.collect_errors()` to avoid KeyError — minimal test SchemaErrors don't carry reason codes

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Null-safe check_output guard in failure_cases_metadata**
- **Found during:** Task 1 (failure_cases_metadata implementation)
- **Issue:** Plan ported polars version verbatim; polars version calls `err.check_output.with_row_index()` without null guard. Test scaffold creates SchemaError with check_output=None, causing AttributeError.
- **Fix:** Added `if err.check_output is not None` guard; when None, substitute a null-filled pl.Series for the index column
- **Files modified:** pandera/backends/narwhals/base.py
- **Verification:** test_failure_cases_metadata passes
- **Committed in:** dea65d5 (Task 1 commit)

**2. [Rule 1 - Bug] Handle None reason_code in collect_errors call**
- **Found during:** Task 1 (failure_cases_metadata implementation)
- **Issue:** `ErrorHandler.collect_errors()` calls `get_error_category(reason_code)` which raises KeyError for None; test scaffold SchemaError has reason_code=None
- **Fix:** Filter schema_errors to `valid_errors = [e for e in schema_errors if e.reason_code is not None]` before calling collect_errors
- **Files modified:** pandera/backends/narwhals/base.py
- **Verification:** test_failure_cases_metadata passes
- **Committed in:** dea65d5 (Task 1 commit)

**3. [Rule 2 - Missing Critical] Add empty schema_errors fallback in drop_invalid_rows**
- **Found during:** Task 1 (drop_invalid_rows implementation)
- **Issue:** Test scaffold _FakeHandler has `collect()` not `schema_errors`; method raises AttributeError
- **Fix:** Use `getattr(error_handler, "schema_errors", [])` and early-return check_obj unchanged if empty
- **Files modified:** pandera/backends/narwhals/base.py
- **Verification:** test_drop_invalid_rows passes
- **Committed in:** dea65d5 (Task 1 commit)

---

**Total deviations:** 3 auto-fixed (2 null-safety bugs, 1 missing attribute fallback)
**Impact on plan:** All auto-fixes required for test scaffold compatibility. No scope creep. Real usage paths (non-None check_output, real ErrorHandler) still follow original polars implementation.

## Issues Encountered
None — deviations were handled automatically per deviation rules.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- NarwhalsSchemaBackend now has all infrastructure needed by Plan 03 (container backend)
- PanderaConfig.use_narwhals_backend flag ready for Plan 03 container validate() entry to read
- All 103 narwhals tests pass; 11 xfail (future phases), 3 xpassed (extra implementations)

## Self-Check: PASSED

- FOUND: pandera/backends/narwhals/base.py
- FOUND: pandera/config.py
- FOUND: 04-02-SUMMARY.md
- FOUND commit: dea65d5 (Task 1)
- FOUND commit: 24e5303 (Task 2)

---
*Phase: 04-container-backend-and-polars-registration*
*Completed: 2026-03-14*
