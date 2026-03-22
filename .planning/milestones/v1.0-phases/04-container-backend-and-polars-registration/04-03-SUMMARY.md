---
phase: 04-container-backend-and-polars-registration
plan: "03"
subsystem: backend
tags: [narwhals, polars, container, validation, lazy, strict, schema-components]

# Dependency graph
requires:
  - phase: 04-container-backend-and-polars-registration-02
    provides: NarwhalsSchemaBackend with failure_cases_metadata, drop_invalid_rows, use_narwhals_backend config flag
  - phase: 03-column-backend-02
    provides: ColumnBackend with check_nullable, check_unique, check_dtype, run_checks

provides:
  - DataFrameSchemaBackend in pandera/backends/narwhals/container.py implementing full validate() pipeline
  - register_narwhals_backends() in pandera/backends/narwhals/register.py (idempotent via lru_cache)
  - ColumnBackend.validate() method enabling column-level validation dispatch from container
  - schema.validate(pl.DataFrame) returns pl.DataFrame (type-preserving)
  - schema.validate(pl.LazyFrame) returns pl.LazyFrame (type-preserving)
  - strict=True raises SchemaError for extra columns; strict="filter" drops them
  - lazy=True collects all errors before raising SchemaErrors
  - SchemaError.failure_cases is always native pl.DataFrame

affects:
  - 04-04 (Polars registration uses register_narwhals_backends)
  - Phase 5 (Ibis) will extend validate() and column components

# Tech tracking
tech-stack:
  added: []
  patterns:
    - _to_lazy_nw / _to_frame_kind_nw helpers for narwhals LazyFrame wrapping/unwrapping
    - return_type = type(check_obj) captured at validate() entry for type-preserving returns
    - Phase 4 parsers list contains ONLY strict_filter_columns (coerce/default deferred)
    - Native frame passed to schema_component.validate() for correct backend dispatch
    - SchemaErrors.data uses _to_native(check_lf) to avoid BackendNotFoundError

key-files:
  created:
    - pandera/backends/narwhals/container.py
    - pandera/backends/narwhals/register.py
  modified:
    - pandera/backends/narwhals/components.py

key-decisions:
  - "Native pl.LazyFrame passed to schema_component.validate() — Column backend registered for pl.LazyFrame not nw.LazyFrame"
  - "Check registered for nw.LazyFrame and nw.DataFrame so run_check dispatches when ColumnBackend wraps to narwhals"
  - "check_dtype in ColumnBackend uses two-pass strategy: narwhals engine first, native polars dtype fallback — enables polars_engine dtypes (pl.Int64) in Column schemas"
  - "SchemaErrors.data uses _to_native() to prevent BackendNotFoundError in SchemaErrors.__init__ when data is nw.LazyFrame"
  - "ColumnBackend.validate() added to components.py — required for run_schema_component_checks dispatch path"

patterns-established:
  - "Narwhals container wraps to nw.LazyFrame for parsers/checks, passes native frames to Column.validate()"
  - "Two-pass dtype check: narwhals_engine.Engine.dtype() first, native polars schema fallback for polars_engine schemas"
  - "Register Check backend for both nw.LazyFrame and pl.LazyFrame in register_narwhals_backends()"

requirements-completed:
  - CONTAINER-02
  - CONTAINER-03
  - CONTAINER-04

# Metrics
duration: 8min
completed: 2026-03-14
---

# Phase 4 Plan 03: DataFrameSchemaBackend Container Implementation Summary

**DataFrameSchemaBackend with full validate() pipeline — type-preserving Polars DataFrame/LazyFrame validation with strict column modes and lazy error collection**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-14T13:27:59Z
- **Completed:** 2026-03-14T13:36:00Z
- **Tasks:** 1
- **Files modified:** 3

## Accomplishments
- DataFrameSchemaBackend.validate() preserves input type (pl.DataFrame in → pl.DataFrame out, pl.LazyFrame in → pl.LazyFrame out)
- strict=True raises SchemaError for extra columns; strict="filter" silently drops them
- lazy=True collects all column check errors before raising SchemaErrors (multi-error mode)
- SchemaError.failure_cases is always a native pl.DataFrame (not nw.DataFrame wrapper)
- register_narwhals_backends() is idempotent via lru_cache
- All 7 targeted CONTAINER-02/03/04 xfail tests promoted to XPASS

## Task Commits

Each task was committed atomically:

1. **Task 1: Create DataFrameSchemaBackend in container.py with validate(), strict_filter_columns, and helper methods** - `dd9f2cd` (feat)

**Plan metadata:** (docs commit to follow)

## Files Created/Modified
- `pandera/backends/narwhals/container.py` - New file: DataFrameSchemaBackend with validate(), collect_column_info(), collect_schema_components(), strict_filter_columns(), check_column_presence(), check_column_values_are_unique(), run_schema_component_checks(), run_checks(); module-level _to_lazy_nw() and _to_frame_kind_nw() helpers
- `pandera/backends/narwhals/register.py` - New file: register_narwhals_backends() with lru_cache; registers DataFrameSchemaBackend for pl.DataFrame/pl.LazyFrame, ColumnBackend for pl.LazyFrame, NarwhalsCheckBackend for pl.LazyFrame/nw.LazyFrame/nw.DataFrame
- `pandera/backends/narwhals/components.py` - Added ColumnBackend.validate() method, get_regex_columns() method; fixed check_dtype() to support polars_engine schema dtypes via native polars schema fallback

## Decisions Made
- Native `pl.LazyFrame` passed to `schema_component.validate()` via `_to_native()` — Column backend is registered for `pl.LazyFrame`, not `nw.LazyFrame`, and Column.validate() dispatch uses the native type
- `Check` registered for `nw.LazyFrame` and `nw.DataFrame` in register_narwhals_backends() — ColumnBackend.validate() wraps to narwhals internally, so run_check() sees a nw.LazyFrame and needs a Check backend for that type
- `check_dtype` two-pass strategy: try `narwhals_engine.Engine.dtype(nw_dtype)` first (for narwhals_engine schemas), then fall back to native polars dtype (for polars_engine Column schemas like `Column(pl.Int64)`)
- `SchemaErrors(data=_to_native(check_lf))` in ColumnBackend.validate() — SchemaErrors.__init__ calls `schema.get_backend(data)` which fails for nw.LazyFrame

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added ColumnBackend.validate() to components.py**
- **Found during:** Task 1 (run_schema_component_checks execution)
- **Issue:** ColumnBackend inherited BaseSchemaBackend.validate() which raises NotImplementedError. Container's run_schema_component_checks calls schema_component.validate(check_obj), dispatching to ColumnBackend.validate() which was not implemented.
- **Fix:** Added validate() method to ColumnBackend mirroring polars ColumnBackend.validate() but using narwhals frame wrapping; also added get_regex_columns() for regex column support
- **Files modified:** pandera/backends/narwhals/components.py
- **Verification:** test_validate_polars_dataframe, test_validate_polars_lazyframe pass
- **Committed in:** dd9f2cd (Task 1 commit)

**2. [Rule 1 - Bug] Fixed check_dtype to handle polars_engine schema dtypes**
- **Found during:** Task 1 (dtype check for Column(pl.Int64) schema)
- **Issue:** narwhals ColumnBackend.check_dtype() converted nw_dtype via narwhals_engine.Engine.dtype(), returning narwhals_engine.Int64. But Column(pl.Int64) schema has polars_engine.Int64 dtype. polars_engine.Int64.check(narwhals_engine.Int64) returns False, causing false dtype failures.
- **Fix:** Two-pass strategy: try narwhals engine first, then fall back to native polars dtype from collect_schema() for polars_engine dtypes; ibis path returns None for native_schema and uses original result
- **Files modified:** pandera/backends/narwhals/components.py
- **Verification:** test_validate_polars_dataframe passes; test_check_dtype_correct[ibis] still passes
- **Committed in:** dd9f2cd (Task 1 commit)

**3. [Rule 3 - Blocking] Register Check for nw.LazyFrame and nw.DataFrame**
- **Found during:** Task 1 (lazy=True error collection path)
- **Issue:** ColumnBackend.validate() wraps check_obj to nw.LazyFrame internally. run_checks_and_handle_errors calls run_check(check_lf, ...) with nw.LazyFrame. Check.get_backend(nw.LazyFrame) raises BackendNotFoundError since Check was only registered for pl.LazyFrame.
- **Fix:** Added Check.register_backend(nw.LazyFrame, NarwhalsCheckBackend) and Check.register_backend(nw.DataFrame, NarwhalsCheckBackend) in register_narwhals_backends()
- **Files modified:** pandera/backends/narwhals/register.py
- **Verification:** test_lazy_mode_collects_all_errors passes with 2 errors collected
- **Committed in:** dd9f2cd (Task 1 commit)

**4. [Rule 1 - Bug] Fixed SchemaErrors data arg in ColumnBackend.validate()**
- **Found during:** Task 1 (lazy=True SchemaErrors creation)
- **Issue:** SchemaErrors.__init__ calls schema.get_backend(data) to look up backend. Passing check_lf (nw.LazyFrame) caused BackendNotFoundError since Column is only registered for pl.LazyFrame.
- **Fix:** Pass _to_native(check_lf) as data in SchemaErrors constructor — returns native pl.LazyFrame which has a registered backend
- **Files modified:** pandera/backends/narwhals/components.py
- **Verification:** test_lazy_mode_collects_all_errors passes
- **Committed in:** dd9f2cd (Task 1 commit)

---

**Total deviations:** 4 auto-fixed (1 missing method, 2 blocking backend dispatch issues, 1 dtype check bug)
**Impact on plan:** All auto-fixes required for correct dispatch through the validate() pipeline. No scope creep. The deviations reveal the coupling between narwhals-wrapped frames and native frame backend registration.

## Issues Encountered
None — all issues handled automatically via deviation rules.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- DataFrameSchemaBackend fully functional for Polars DataFrame and LazyFrame validation
- register_narwhals_backends() ready for Plan 04 to wire into PanderaConfig opt-in path
- All 103 narwhals tests pass; no regressions introduced

## Self-Check: PASSED

- FOUND: pandera/backends/narwhals/container.py
- FOUND: pandera/backends/narwhals/register.py
- FOUND: pandera/backends/narwhals/components.py (modified)
- FOUND commit: dd9f2cd (Task 1)
- FOUND: 04-03-SUMMARY.md

---
*Phase: 04-container-backend-and-polars-registration*
*Completed: 2026-03-14*
