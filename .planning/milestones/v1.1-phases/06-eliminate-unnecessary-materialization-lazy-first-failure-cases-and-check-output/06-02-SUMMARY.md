---
phase: 06-eliminate-unnecessary-materialization-lazy-first-failure-cases-and-check-output
plan: "02"
subsystem: narwhals
tags: [narwhals, polars, ibis, lazy-frame, failure-cases, materialization, check-nullable]

# Dependency graph
requires:
  - phase: 06-eliminate-unnecessary-materialization-lazy-first-failure-cases-and-check-output
    plan: "01"
    provides: RED baseline tests for subsample() lazy-first and failure_cases type contracts
provides:
  - Unified run_check() with no _is_ibis_result bifurcation — single 40-line code path
  - subsample() delegates directly to .head()/.tail() — stays lazy for polars and ibis-safe for tail=
  - check_nullable() materializes only the scalar .any() — full frame never collected
  - SchemaError.failure_cases is now native (pl.DataFrame for polars, ibis.Table for ibis)
  - failure_cases_metadata() handles native ibis.Table inputs (wraps to narwhals for unified processing)
  - NarwhalsErrorHandler._count_failure_cases handles ibis.Table via .count().execute()
affects:
  - 06-03 (failure_cases_metadata redesign + ibis lazy SchemaErrors contract)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "LazyFrame backend detection: hasattr(nw.to_native(lf), 'execute') distinguishes ibis from polars"
    - "Native conversion: nw.LazyFrame ibis → nw.to_native() directly; polars → _materialize() then to_native()"
    - "Scalar-only materialization: _materialize(combined_lf.select(nw.col(KEY).any())) materializes one row"
    - "Ibis.Table in failure_cases_metadata: wrap back to nw.from_native() to reuse narwhals materialization path"

key-files:
  created: []
  modified:
    - pandera/backends/narwhals/base.py
    - pandera/backends/narwhals/components.py
    - pandera/api/narwhals/error_handler.py

key-decisions:
  - "ibis nw.LazyFrame failure_cases: nw.to_native(lf) gives ibis.Table without execution — use hasattr(native, 'execute') to detect and skip _materialize()"
  - "failure_cases_metadata handles native ibis.Table: wrap to nw.from_native() then reuse existing narwhals materialization path — avoids duplicating the pl.from_arrow conversion logic"
  - "NarwhalsErrorHandler._count_failure_cases: ibis.Table.count().execute() is the correct ibis count — len() raises ExpressionError"
  - "check_nullable check_output stays as full combined_lf (wide frame with CHECK_OUTPUT_KEY column) — failure_cases_metadata uses this for index computation"

patterns-established:
  - "Backend-agnostic native check: hasattr(nw.to_native(frame), 'execute') identifies SQL-lazy backends"
  - "Narwhals-to-native conversion: always check LazyFrame vs DataFrame type before deciding collect vs direct to_native"

requirements-completed:
  - LAZY-FIRST-01

# Metrics
duration: 25min
completed: 2026-03-23
---

# Phase 6 Plan 02: Lazy-First run_check() + check_nullable() Summary

**Collapsed dead _is_ibis_result bifurcation in run_check(), removed fc.collect(), and fixed check_nullable() to materialize only the scalar .any() — SchemaError.failure_cases is now native (pl.DataFrame / ibis.Table) throughout**

## Performance

- **Duration:** 25 min
- **Started:** 2026-03-23T17:55:00Z
- **Completed:** 2026-03-23T22:17:52Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- `run_check()` in base.py now has a single ~35-line unified code path: no `_is_ibis_result` detection, no `fc.collect()`, no `_materialize(check_output)` — only `_materialize(passed_lf)[CHECK_OUTPUT_KEY][0]` for the scalar bool
- `subsample()` in base.py delegates directly to `.head()/.tail()` without `_materialize(check_obj)` — stays `nw.LazyFrame` for polars, raises `NotImplementedError` for ibis tail= (detected via `hasattr(native, "execute")`)
- `check_nullable()` in components.py uses `_materialize(combined_lf.select(nw.col(CHECK_OUTPUT_KEY).any()))` — one row materialized, not the full frame
- `run_checks_and_handle_errors()` in components.py converts narwhals wrappers to native before setting `SchemaError.failure_cases`: polars `nw.LazyFrame` → collect + `to_native` → `pl.DataFrame`; ibis `nw.LazyFrame` → `nw.to_native()` directly → `ibis.Table`
- 8 previously RED tests now GREEN: 5 TestSubsample + 3 TestBuiltinChecks e2e tests (polars and ibis failure_cases native type contracts)
- `test_failure_cases_is_native` (test_container.py) and `test_failure_cases_native_ibis` (test_parity.py) also turned GREEN as a result

## Task Commits

Each task was committed atomically:

1. **Task 1: Rewrite subsample() and run_check() in base.py** - `d6e3efa` (feat)
2. **Task 2: Fix check_nullable() + failure_cases native conversion** - `007a6e3` (feat)

**Plan metadata:** (docs commit — see below)

## Files Created/Modified

- `pandera/backends/narwhals/base.py` — subsample() lazy-first; run_check() unified; failure_cases_metadata() handles native ibis.Table
- `pandera/backends/narwhals/components.py` — check_nullable() scalar-only; run_checks_and_handle_errors() converts failure_cases to native
- `pandera/api/narwhals/error_handler.py` — _count_failure_cases handles ibis.Table via .count().execute()

## Decisions Made

- **ibis nw.LazyFrame native conversion:** `nw.to_native(lf)` on an ibis-backed `nw.LazyFrame` returns `ibis.Table` directly (no execution). `hasattr(native, "execute")` cleanly distinguishes ibis from polars. Using `_materialize()` first would execute the ibis query and return `pyarrow.Table` — incorrect.
- **failure_cases_metadata + ibis.Table:** when `SchemaError.failure_cases` is native `ibis.Table`, wrap it back to narwhals via `nw.from_native()` to reuse the existing `_materialize() → to_arrow() → pl.from_arrow()` path. Avoids duplicating conversion logic.
- **_count_failure_cases for ibis:** `ibis.Table.__len__()` raises `ExpressionError: Use .count() instead`. Added guarded ibis import to `NarwhalsErrorHandler._count_failure_cases` with `ibis.Table.count().execute()`.
- **check_output stays as full combined_lf:** the wide frame with CHECK_OUTPUT_KEY column is passed as check_output in check_nullable(). This is used by failure_cases_metadata() for index computation when CHECK_OUTPUT_KEY is present.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Handle native ibis.Table in failure_cases_metadata()**
- **Found during:** Task 2 (failure_cases native conversion)
- **Issue:** After converting `SchemaError.failure_cases` to native `ibis.Table` in `run_checks_and_handle_errors`, `failure_cases_metadata()` only handled `(nw.LazyFrame, nw.DataFrame)` and scalars. Passing native `ibis.Table` fell to the scalar branch which tried to serialize `ibis.Table` as a string — crashing in `pl.DataFrame` construction.
- **Fix:** Added a pre-detection block in `failure_cases_metadata()` that wraps native `ibis.Table` back to narwhals with `nw.from_native()` before the existing materialization path.
- **Files modified:** `pandera/backends/narwhals/base.py`
- **Verification:** `test_ibis_lazy_collects_multiple_errors` and `test_lazy_failure_cases_is_dataframe` remain GREEN after fix
- **Committed in:** `007a6e3` (Task 2 commit)

**2. [Rule 2 - Missing Critical] Fix NarwhalsErrorHandler._count_failure_cases for ibis.Table**
- **Found during:** Task 2 (after failure_cases_metadata fix)
- **Issue:** `error_handler.collect_error()` calls `_count_failure_cases(schema_error.failure_cases)`. With failure_cases as native `ibis.Table`, the base `_count_failure_cases` called `len(ibis_table)` which raises `ibis.common.exceptions.ExpressionError: Use .count() instead`.
- **Fix:** Added ibis.Table branch to `NarwhalsErrorHandler._count_failure_cases` using `ibis.Table.count().execute()`.
- **Files modified:** `pandera/api/narwhals/error_handler.py`
- **Verification:** `test_ibis_lazy_collects_multiple_errors` GREEN (was crashing before fix)
- **Committed in:** `007a6e3` (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (2 missing critical — same-task cascading fixes)
**Impact on plan:** Both fixes required for correctness when failure_cases is native ibis.Table in the error pipeline. No scope creep — both fixes are direct consequences of converting failure_cases to native.

## Issues Encountered

- **ibis nw.LazyFrame vs polars nw.LazyFrame distinction:** both are `nw.LazyFrame` type, but `_materialize()` behaves differently — polars collects to `pl.DataFrame`, ibis executes to `pyarrow.Table`. Used `hasattr(nw.to_native(lf), "execute")` to detect ibis backend and skip materialization.
- **validate() forces ibis nw.DataFrame to LazyFrame:** `components.py::validate()` applies `.lazy()` to the input frame (line 63), converting ibis `nw.DataFrame` to `nw.LazyFrame`. So failure_cases from run_check() for ibis inputs come back as `nw.LazyFrame` (not `nw.DataFrame`). The backend detection in `run_checks_and_handle_errors` handles this correctly via `hasattr(native, "execute")`.

## Remaining RED Tests (Out of Scope)

- `test_failure_cases_metadata_ibis_returns_ibis_table` — Plan 03 scope: `FailureCaseMetadata.failure_cases` must be `ibis.Table` for ibis inputs (failure_cases_metadata redesign)
- `test_ibis_lazy_failure_cases_is_ibis_table` — Plan 03 scope: `SchemaErrors.failure_cases` (from failure_cases_metadata) must be `ibis.Table` for ibis lazy validation
- `test_custom_check_receives_table_and_key` — Pre-existing failure: ibis API changed `DatabaseTable` → `Table` class name. Deferred per Plan 01 scope boundary.

## Next Phase Readiness

- Plan 03 (failure_cases_metadata redesign) acceptance criteria: `TestSubsample` and `test_failure_cases_metadata_ibis_returns_ibis_table` in test_components.py — both still RED
- The ibis detection pattern (`hasattr(native, "execute")`) established here is reusable in Plan 03 for failure_cases_metadata ibis path

---
*Phase: 06-eliminate-unnecessary-materialization-lazy-first-failure-cases-and-check-output*
*Completed: 2026-03-23*
