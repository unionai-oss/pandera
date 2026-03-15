---
phase: 05-ibis-registration-and-integration
plan: "02"
subsystem: backend
tags: [narwhals, ibis, polars, registration, lru_cache, group_by, SQL-lazy]

# Dependency graph
requires:
  - phase: 05-ibis-registration-and-integration-01
    provides: test stubs and xfail markers for ibis registration

provides:
  - "@lru_cache + narwhals auto-detection in register_ibis_backends()"
  - "SQL-lazy-safe check_unique via group_by().agg(nw.len()) in components.py"
  - "SQL-lazy-safe check_column_values_are_unique via group_by().agg(nw.len()) in container.py"
  - "ibis-native third dtype pass in check_dtype in components.py"
  - "drop_invalid_rows with nw.all_horizontal() for Polars and IbisSchemaBackend delegation for ibis"
  - "Dead narwhals/register.py deleted"

affects:
  - phase 05 remaining plans
  - any ibis validate() path

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "lru_cache on register_ibis_backends() mirrors polars register pattern"
    - "group_by().agg(nw.len()) as SQL-lazy-safe alternative to is_duplicated()"
    - "three-pass dtype checking: narwhals_engine -> native polars schema -> ibis native schema"
    - "nw.all_horizontal() for boolean reduction instead of pl.fold"
    - "ibis drop_invalid_rows delegated to IbisSchemaBackend, then result wrapped back to narwhals"

key-files:
  created: []
  modified:
    - pandera/backends/ibis/register.py
    - pandera/backends/narwhals/components.py
    - pandera/backends/narwhals/container.py
    - pandera/backends/narwhals/base.py
  deleted:
    - pandera/backends/narwhals/register.py

key-decisions:
  - "group_by().agg(nw.len()) replaces collect()+is_duplicated() for SQL-lazy backends — no per-row boolean output for uniqueness checks"
  - "ibis drop_invalid_rows delegates to IbisSchemaBackend (has positional-join / row_number logic) then wraps result back to narwhals"
  - "narwhals/register.py deleted — was dead file with no imports, registration now handled by polars/register.py and ibis/register.py"
  - "check_dtype ibis third pass uses ibis_table.schema().get(col) — direct ibis schema API, after narwhals and polars passes"

patterns-established:
  - "Pattern: SQL-lazy uniqueness via group_by().agg(nw.len()) — use this pattern for any uniqueness check that must work on ibis/DuckDB"
  - "Pattern: three-pass dtype checking with ibis fallback — narwhals_engine -> polars native -> ibis native"

requirements-completed:
  - REGISTER-03
  - TEST-02

# Metrics
duration: 15min
completed: 2026-03-14
---

# Phase 05 Plan 02: Ibis Backend Core Implementation Summary

**register_ibis_backends() now @lru_cache with narwhals auto-detection; SQL-lazy-safe group_by uniqueness checks; ibis dtype third pass; drop_invalid_rows delegates to IbisSchemaBackend for ibis path**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-03-14T00:00:00Z
- **Completed:** 2026-03-14T00:15:00Z
- **Tasks:** 2
- **Files modified:** 4 (+ 1 deleted)

## Accomplishments

- `register_ibis_backends()` now mirrors the polars register pattern: `@lru_cache`, `check_cls_fqn` parameter, narwhals auto-detection with `UserWarning`, fallback to native ibis backends
- Deleted dead `pandera/backends/narwhals/register.py` — verified no imports anywhere before deletion
- `check_unique` in components.py uses `group_by().agg(nw.len())` — SQL-lazy safe, works on ibis without full materialization
- `check_column_values_are_unique` in container.py uses same `group_by` pattern, imports `_materialize` from base
- `check_dtype` in components.py adds ibis-native third pass via `ibis_table.schema().get(col)` — handles ibis dtypes after narwhals and polars passes fail
- `drop_invalid_rows` in base.py detects ibis path and delegates to `IbisSchemaBackend().drop_invalid_rows()`, then wraps result back to narwhals; Polars path uses `nw.all_horizontal()` replacing `pl.fold`
- All 115 narwhals tests pass; `test_ibis_narwhals_auto_activated` now XPASS (was xfail)

## Task Commits

Each task was committed atomically:

1. **Task 1: Add @lru_cache + narwhals auto-detection to register_ibis_backends()** - `5a96541` (feat)
2. **Task 2: Fix check_unique, check_dtype, check_column_values_are_unique, drop_invalid_rows** - `3807a33` (feat)

## Files Created/Modified

- `pandera/backends/ibis/register.py` - Rewritten: @lru_cache, narwhals auto-detection, UserWarning, fallback to native ibis
- `pandera/backends/narwhals/components.py` - group_by uniqueness + ibis dtype third pass
- `pandera/backends/narwhals/container.py` - SQL-lazy-safe check_column_values_are_unique + _materialize import
- `pandera/backends/narwhals/base.py` - nw.all_horizontal Polars path + IbisSchemaBackend delegation
- `pandera/backends/narwhals/register.py` - DELETED (dead file)

## Decisions Made

- `group_by().agg(nw.len())` replaces `collect()+is_duplicated()` for both column and container uniqueness — the group_by approach is SQL-lazy safe and produces duplicate values rather than per-row booleans (check_output=None for uniqueness failures)
- `drop_invalid_rows` for ibis delegates to `IbisSchemaBackend` which already has the positional-join / row_number logic; result is wrapped back to narwhals via `nw.from_native()`
- ibis dtype third pass is a try/except ImportError guard so it is zero-cost when ibis is absent

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Ibis registration now routes through narwhals backend — `schema.validate(ibis_table)` dispatches to narwhals `DataFrameSchemaBackend`
- SQL-lazy-safe uniqueness checks enable ibis validation without full materialization
- drop_invalid_rows ibis path is wired — enables lazy validation with `drop_invalid_rows=True` for ibis
- Remaining xfail tests: `test_failure_cases_native_ibis` and `test_coerce_ibis` (coerce is intentionally deferred, strict=True gate)

---
*Phase: 05-ibis-registration-and-integration*
*Completed: 2026-03-14*
