---
phase: 05-ibis-registration-and-integration
plan: "03"
subsystem: testing
tags: [narwhals, ibis, polars, registration, test-parity, xfail, element_wise, failure_cases]

# Dependency graph
requires:
  - phase: 05-ibis-registration-and-integration-02
    provides: "@lru_cache on register_ibis_backends() with narwhals auto-detection"

provides:
  - "register_ibis_backends() call in narwhals autouse fixture — conftest.py"
  - "test_parity.py with all ibis tests passing (no xfail except coerce strict=True)"
  - "test_failure_cases_native_ibis: verifies failure_cases is native (pyarrow/pandas/polars) not narwhals wrapper"
  - "test_element_wise_check_raises_not_implemented_ibis: TEST-02 element_wise on ibis raises SchemaError"

affects:
  - any future narwhals test suite changes
  - phase 05 TEST-02 and TEST-04 requirements

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "ibis DuckDB backend materializes via narwhals LazyFrame.collect() → pyarrow.Table (not pandas)"
    - "element_wise=True on SQL-lazy backends raises NotImplementedError wrapped in SchemaError by run_checks"
    - "test_failure_cases_native_ibis accepts pyarrow.Table as valid native type for ibis validation output"

key-files:
  created: []
  modified:
    - tests/backends/narwhals/conftest.py
    - tests/backends/narwhals/test_parity.py

key-decisions:
  - "failure_cases for ibis validation is pyarrow.Table (not pandas) — ibis DuckDB backend returns pyarrow when narwhals LazyFrame.collect() is called; test updated to accept pyarrow as valid native type"
  - "element_wise=True SchemaError wraps NotImplementedError via run_checks exception catch — test asserts SchemaError raised with NotImplementedError in message"

patterns-established:
  - "Pattern: register_ibis_backends() + register_polars_backends() both called in narwhals autouse fixture — both .cache_clear() + call"
  - "Pattern: ibis failure_cases native type is pyarrow.Table — test native type assertions must include pyarrow"

requirements-completed:
  - TEST-02
  - TEST-04

# Metrics
duration: 15min
completed: 2026-03-15
---

# Phase 05 Plan 03: Ibis Test Suite Wiring Summary

**conftest.py wired with register_ibis_backends(); test_parity.py converted from 7 xfail stubs to passing tests; pyarrow materialization behavior documented and test_failure_cases_native_ibis updated accordingly**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-03-15T05:00:00Z
- **Completed:** 2026-03-15T05:15:00Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- `conftest.py` autouse fixture now calls `register_ibis_backends.cache_clear()` and `register_ibis_backends()` alongside the polars equivalents — narwhals + ibis backends both active for all narwhals tests
- Removed xfail marks from 7 ibis parity tests that all xpassed after ibis registration was wired (Plan 02 had already landed the implementation)
- Updated `test_failure_cases_native_ibis`: discovered that ibis DuckDB backend materializes to `pyarrow.Table` (not `pandas.DataFrame`) when going through the narwhals `LazyFrame.collect()` path; test now accepts pyarrow as valid native type
- Added `test_element_wise_check_raises_not_implemented_ibis` (TEST-02): verifies element_wise checks on ibis raise `SchemaError` wrapping `NotImplementedError`; the inner `NotImplementedError` from `checks.py` is captured by `run_checks` and surfaced as a `SchemaError`
- `test_coerce_ibis` preserved as `xfail(strict=True)` — v2 coerce feature gate, unchanged
- Full narwhals test suite: 124 passed, 1 skipped, 3 xfailed, 4 xpassed (xpasses are from test_container.py stubs, strict=False, unrelated to this plan)

## Task Commits

Each task was committed atomically:

1. **Task 1: Update conftest.py to register ibis backends in autouse fixture** - `cfea149` (feat)
2. **Task 2: Convert test_parity.py stubs to passing tests + verify TEST-02 ibis coverage** - `c9f8d8d` (feat)

## Files Created/Modified

- `tests/backends/narwhals/conftest.py` - Added register_ibis_backends import + cache_clear() + call in _suppress_narwhals_warning fixture
- `tests/backends/narwhals/test_parity.py` - Removed xfail from 7 passing tests; updated failure_cases assertion to accept pyarrow; added element_wise NotImplementedError test

## Decisions Made

- `failure_cases` for ibis validation is `pyarrow.lib.Table` not `pandas.DataFrame`: when ibis-backed narwhals DataFrame is converted to LazyFrame via `.lazy()` and then `collect()` is called, the result stays in ibis/pyarrow domain. `nw.to_native()` then returns pyarrow. The test was updated to accept `pyarrow.Table` alongside `pd.DataFrame` and `pl.DataFrame` as valid native types.
- `element_wise=True` on ibis raises `SchemaError` (not `NotImplementedError` directly): the inner `NotImplementedError` from `checks.py:apply()` is caught by `run_checks` and converted to a `CoreCheckResult` with `CHECK_ERROR` reason code, which becomes a `SchemaError`. The test asserts `SchemaError` and checks the message contains "NotImplementedError".

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] test_failure_cases_native_ibis assertion updated to accept pyarrow.Table**
- **Found during:** Task 2 (converting xfail stubs to passing tests)
- **Issue:** The test asserted `isinstance(fc, (pd.DataFrame, pl.DataFrame))` but the actual native type from ibis DuckDB via narwhals LazyFrame.collect() is `pyarrow.lib.Table`
- **Fix:** Added `pyarrow.Table` to the accepted types tuple, with try/except ImportError guard for environments without pyarrow
- **Files modified:** tests/backends/narwhals/test_parity.py
- **Verification:** test_failure_cases_native_ibis now passes
- **Committed in:** c9f8d8d (Task 2 commit)

**2. [Rule 1 - Bug] element_wise test uses SchemaError not NotImplementedError**
- **Found during:** Task 2 (adding element_wise test)
- **Issue:** Plan specified `pytest.raises(NotImplementedError)` but run_checks captures all exceptions and wraps them in SchemaError with CHECK_ERROR reason_code
- **Fix:** Test asserts `pytest.raises(SchemaError)` and checks message contains "NotImplementedError" or "element_wise"
- **Files modified:** tests/backends/narwhals/test_parity.py
- **Verification:** test_element_wise_check_raises_not_implemented_ibis passes
- **Committed in:** c9f8d8d (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (both Rule 1 - Bug)
**Impact on plan:** Both fixes correct the test assertions to match actual behavior. No scope creep. Tests correctly document the ibis backend's materialization and error-wrapping behavior.

## Issues Encountered

- Pre-existing failures in `tests/ibis/` test suite (103 failures) from Plan 02's narwhals auto-detection in `register_ibis_backends()` routing ibis tests through the narwhals backend. These failures pre-existed before any Plan 03 changes and are out of scope for this plan.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- TEST-02 and TEST-04 requirements are met: ibis coverage for all validation behaviors verified, parity suite green for both Polars and Ibis
- The 4 xpassing tests in test_container.py (test_failure_cases_metadata, test_drop_invalid_rows, test_ibis_narwhals_auto_activated, test_ibis_backend_is_narwhals) are ready to have their xfail marks removed in a cleanup plan
- Pre-existing `tests/ibis/` failures from the narwhals backend routing need investigation in a future plan

---
*Phase: 05-ibis-registration-and-integration*
*Completed: 2026-03-15*
