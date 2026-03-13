# Roadmap: Pandera Narwhals Backend

## Overview

Build a Narwhals-backed validation backend for pandera in five ordered phases, each delivering a coherent layer of the implementation. The build order is constrained by hard dependencies: utility types and the dtype engine must exist before any check or coercion code is written; the check backend must exist before the column backend; the column backend must exist before the container backend; and all backends must be complete before registration can be tested end-to-end. Polars validation runs end-to-end at Phase 4; Ibis validation closes the known xfail gaps at Phase 5.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Foundation** - Utility types, narwhals dtype engine, and dependency scaffolding (completed 2026-03-09)
- [x] **Phase 2: Check Backend** - NarwhalsCheckBackend, all 14 builtin checks, and initial test harness (completed 2026-03-10)
- [ ] **Phase 3: Column Backend** - Per-column nullable/unique/dtype/run_checks pipeline
- [ ] **Phase 4: Container Backend and Polars Registration** - Full validation pipeline, end-to-end Polars validation green
- [ ] **Phase 5: Ibis Registration and Integration** - Ibis end-to-end, xfail gap closure, full test suite

## Phase Details

### Phase 1: Foundation
**Goal**: The dtype engine, utility types, and registration scaffolding exist and are correct — every subsequent phase can build on them without retrofitting
**Depends on**: Nothing (first phase)
**Requirements**: INFRA-01, INFRA-02, INFRA-03, ENGINE-01, ENGINE-02, ENGINE-03
**Success Criteria** (what must be TRUE):
  1. `narwhals>=2.15.0` is installable as `pandera[narwhals]` and all backend code imports from `narwhals.stable.v1`
  2. `NarwhalsData` named tuple exists and the `Dispatcher` can route to it (analogous to `PolarsData`)
  3. `narwhals_engine.py` exists with all core dtypes registered (`nw.Int8/16/32/64`, `nw.UInt*`, `nw.Float32/64`, `nw.String`, `nw.Boolean`, `nw.Date`, `nw.Datetime`, `nw.Duration`, `nw.Categorical`, `nw.List`, `nw.Struct`)
  4. `coerce()` and `try_coerce()` work via `nw.col(name).cast(nw_dtype)` and return native frames
  5. `_to_native()` helper exists and is used at every `SchemaError` construction site
**Plans**: 2 plans

Plans:
- [ ] 01-01-PLAN.md — pyproject.toml narwhals extra, `pandera/api/narwhals/` package (types, utils), and test scaffold
- [ ] 01-02-PLAN.md — `pandera/engines/narwhals_engine.py` with all 11 dtype registrations and coerce/try_coerce

### Phase 2: Check Backend
**Goal**: All 14 builtin checks execute correctly through the narwhals Expr API, and the check dispatch chain is tested end-to-end
**Depends on**: Phase 1
**Requirements**: CHECKS-01, CHECKS-02, CHECKS-03, TEST-01
**Success Criteria** (what must be TRUE):
  1. `NarwhalsCheckBackend` routes builtin checks to `NarwhalsData` containers and user-defined checks to native containers
  2. All 14 builtin checks (`equal_to`, `not_equal_to`, `greater_than`, `greater_than_or_equal_to`, `less_than`, `less_than_or_equal_to`, `in_range`, `isin`, `notin`, `str_matches`, `str_contains`, `str_startswith`, `str_endswith`, `str_length`) pass against a narwhals-wrapped frame
  3. `element_wise=True` checks on SQL-lazy backends raise `NotImplementedError` with a clear message rather than silently failing
  4. `tests/backends/narwhals/` exists with a parameterized test harness runnable against at least one backend
**Plans**: 3 plans

Plans:
- [ ] 02-01-PLAN.md — package init, test conftest with make_narwhals_frame fixture (polars+ibis), and xfail test stubs
- [ ] 02-02-PLAN.md — `NarwhalsCheckBackend` with routing, element_wise guard, and postprocess (CHECKS-01, CHECKS-03)
- [ ] 02-03-PLAN.md — all 14 narwhals builtin checks via Expr API (CHECKS-02)

### Phase 3: Column Backend
**Goal**: Per-column validation (nullable, unique, dtype, run_checks) works correctly and is tested in isolation before being wired into the full container pipeline
**Depends on**: Phase 2
**Requirements**: COLUMN-01, COLUMN-02
**Success Criteria** (what must be TRUE):
  1. `check_nullable` correctly handles both `is_null()` and float `is_nan()` so nullable=False columns with NaN values fail validation
  2. `check_unique` forces `.collect()` before `is_duplicated()` and correctly detects duplicates on both eager and lazy frames
  3. `check_dtype` resolves column dtype through `narwhals_engine` and produces a `SchemaError` with a native frame in `failure_cases`
**Plans**: TBD

### Phase 4: Container Backend and Polars Registration
**Goal**: End-to-end `schema.validate(df)` works for Polars DataFrames and LazyFrames — the full validation pipeline runs, errors surface correctly, and narwhals wrappers never appear in user-visible output
**Depends on**: Phase 3
**Requirements**: CONTAINER-01, CONTAINER-02, CONTAINER-03, CONTAINER-04, REGISTER-01, REGISTER-02, REGISTER-04, TEST-03
**Success Criteria** (what must be TRUE):
  1. `schema.validate(pl.DataFrame(...))` succeeds for a valid frame and raises `SchemaError` for an invalid frame, with `failure_cases` as a native Polars DataFrame (not a narwhals wrapper)
  2. `schema.validate(pl.LazyFrame(...))` works end-to-end and `.collect()` is called only at defined boundaries
  3. `strict=True` and `filter=True` column modes correctly reject or drop unexpected columns
  4. `lazy=True` validation collects all errors before raising `SchemaErrors` (not first-error-only)
  5. Narwhals backend is never registered by default; `pandera.use_backend("narwhals")` or `import pandera.narwhals` is required to activate it
**Plans**: TBD

### Phase 5: Ibis Registration and Integration
**Goal**: End-to-end `schema.validate(table)` works for Ibis Tables, closing all known xfail gaps from the existing Ibis backend, and the full test suite passes against both Polars and Ibis
**Depends on**: Phase 4
**Requirements**: REGISTER-03, TEST-02
**Success Criteria** (what must be TRUE):
  1. `schema.validate(ibis_table)` succeeds for a valid table and raises `SchemaError` for an invalid table
  2. Previously xfailing Ibis backend tests for `coerce_dtype` and column `unique` now pass via the narwhals backend
  3. `element_wise=True` checks on Ibis tables raise `NotImplementedError` with a clear explanation
  4. `SchemaError.failure_cases` on Ibis validation is always a native (non-narwhals) frame type — asserted by tests
  5. All 14 builtin checks, lazy validation, dtype coercion, and error message correctness tests pass against both Polars and Ibis backends
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation | 2/2 | Complete   | 2026-03-09 |
| 2. Check Backend | 3/3 | Complete   | 2026-03-10 |
| 3. Column Backend | 0/TBD | Not started | - |
| 4. Container Backend and Polars Registration | 0/TBD | Not started | - |
| 5. Ibis Registration and Integration | 0/TBD | Not started | - |
