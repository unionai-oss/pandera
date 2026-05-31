---
phase: 06-eliminate-unnecessary-materialization-lazy-first-failure-cases-and-check-output
plan: "01"
subsystem: testing
tags: [narwhals, polars, ibis, lazy-frame, failure-cases, tdd, red-baseline]

# Dependency graph
requires:
  - phase: 05-expression-based-check-protocol-eliminate-framework-specific-apply-branching
    provides: uniform apply() expression protocol; _is_ibis_result dead code ready for removal
provides:
  - RED baseline tests for subsample() lazy-first contracts (head=, tail=, ibis tail raises)
  - RED baseline test for failure_cases_metadata() returning ibis.Table for ibis inputs
  - Updated test_e2e.py assertions for Phase 6 SchemaError.failure_cases type contracts
affects:
  - 06-02 (implements run_check lazy-first, removes _is_ibis_result)
  - 06-03 (implements failure_cases_metadata redesign, subsample fixes)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "RED baseline first: write failing tests before implementation to define acceptance criteria"
    - "SimpleNamespace with custom class via type('Name', (), {}) for schema stubs in unit tests"
    - "ibis_only = pytest.mark.skipif(not HAS_IBIS) guard pattern in test files without ibis fixture"

key-files:
  created: []
  modified:
    - tests/backends/narwhals/test_components.py
    - tests/backends/narwhals/test_e2e.py

key-decisions:
  - "Phase 6 SchemaError.failure_cases contract: native type (pl.DataFrame for polars, ibis.Table for ibis) — not narwhals wrapper"
  - "Phase 6 SchemaErrors.failure_cases contract: ibis.Table for ibis inputs — failure_cases_metadata() must not force pl.DataFrame conversion"
  - "Phase 6 subsample() contract: head= and tail= stay lazy (nw.LazyFrame); ibis tail= raises NotImplementedError"
  - "test_custom_check_receives_table_and_key pre-existing failure (DatabaseTable vs Table ibis API change) deferred — out of scope for Phase 6"

patterns-established:
  - "RED-before-GREEN pattern: test files updated with new contracts before any implementation change"
  - "TestSubsample class structure: polars-only lazy tests + @ibis_only for backend-specific behavior"

requirements-completed:
  - LAZY-FIRST-01

# Metrics
duration: 15min
completed: 2026-03-23
---

# Phase 6 Plan 01: RED Baseline Tests Summary

**RED baseline tests establishing lazy-first failure_cases and subsample() contracts: 9 failing tests (5 in test_components.py + 4 in test_e2e.py) define the Phase 6 acceptance criteria**

## Performance

- **Duration:** 15 min
- **Started:** 2026-03-23T17:35:59Z
- **Completed:** 2026-03-23T17:51:00Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Added `TestSubsample` class with 5 tests covering `head=` stays lazy, `tail=` stays lazy, `head= + tail=` stays lazy, ibis `tail=` raises `NotImplementedError`, and no-params returns unchanged
- Added `test_failure_cases_metadata_ibis_returns_ibis_table` asserting `FailureCaseMetadata.failure_cases` is `ibis.Table` for ibis inputs
- Updated 4 assertions in test_e2e.py to match Phase 6 contracts: polars `SchemaError.failure_cases` is `pl.DataFrame`, ibis `SchemaError.failure_cases` is `ibis.Table`, ibis `SchemaErrors.failure_cases` is `ibis.Table`
- All 9 new/updated tests are correctly RED against the current implementation; 59 existing passing tests remain GREEN

## Task Commits

Each task was committed atomically:

1. **Task 1: Add RED subsample() and failure_cases_metadata() tests to test_components.py** - `7d2e6fd` (test)
2. **Task 2: Update type assertions in test_e2e.py to match Phase 6 contracts** - `df831da` (test)

**Plan metadata:** (docs commit — see below)

## Files Created/Modified

- `tests/backends/narwhals/test_components.py` - Added TestSubsample (5 tests) and test_failure_cases_metadata_ibis_returns_ibis_table; imported NarwhalsSchemaBackend, SchemaError
- `tests/backends/narwhals/test_e2e.py` - Updated 4 test assertions to Phase 6 failure_cases type contracts; renamed test_ibis_lazy_failure_cases_is_dataframe to test_ibis_lazy_failure_cases_is_ibis_table

## Decisions Made

- Phase 6 `SchemaError.failure_cases` contract is **native** (unwrapped from narwhals): `pl.DataFrame` for polars, `ibis.Table` for ibis. This is consistent with the polars backend's behavior and removes the `nw.DataFrame` wrapper that was previously used for ibis inputs.
- Phase 6 `SchemaErrors.failure_cases` contract is also **native** — `failure_cases_metadata()` must preserve the ibis backend type rather than always converting to `pl.DataFrame` via `to_arrow() + pl.from_arrow()`.
- `test_custom_check_receives_table_and_key` is a pre-existing failure (ibis API changed from `DatabaseTable` to `Table` class name) — deferred per scope boundary rules; logged to `deferred-items.md`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed SimpleNamespace schema stub construction for failure_cases_metadata test**
- **Found during:** Task 1 (test_failure_cases_metadata_ibis test)
- **Issue:** Plan suggested `SimpleNamespace(__class__=type("Column", (), {"__name__": "Column"})())` — this fails with `TypeError: cannot set '__name__' attribute of immutable type 'types.SimpleNamespace'`
- **Fix:** Used `type("Column", (), {})` to create a proper class, then instantiated it — `ColumnStub = type("Column", (), {}); schema_stub = ColumnStub(); schema_stub.name = "x"` — `__class__.__name__` is `"Column"` automatically
- **Files modified:** tests/backends/narwhals/test_components.py
- **Verification:** Test setup no longer raises TypeError; test fails with AssertionError (correct RED)
- **Committed in:** `7d2e6fd` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 bug in plan's suggested code pattern)
**Impact on plan:** Minor fix to test construction pattern; no scope change. All new tests correctly RED.

## Issues Encountered

- Pre-existing failure: `test_custom_check_receives_table_and_key` asserts `table_type == "DatabaseTable"` but ibis now returns class name `"Table"`. This was failing before Plan 01 work and is unrelated to Phase 6 scope. Deferred.

## Next Phase Readiness

- Plan 02 (run_check lazy-first: remove fc.collect(), remove _is_ibis_result, remove _materialize(check_output)) — acceptance criteria defined by the 4 RED tests in test_e2e.py
- Plan 03 (failure_cases_metadata redesign + subsample fixes) — acceptance criteria defined by TestSubsample + test_failure_cases_metadata_ibis_returns_ibis_table in test_components.py

---
*Phase: 06-eliminate-unnecessary-materialization-lazy-first-failure-cases-and-check-output*
*Completed: 2026-03-23*
