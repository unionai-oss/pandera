# Roadmap: Pandera Narwhals Backend

## Milestones

- ✅ **v1.0 Narwhals Backend** — Phases 1-5 (shipped 2026-03-15)
- ✅ **v1.1 Ibis Parity & Lazy-First Architecture** — Phases 1-9 (shipped 2026-03-25)
- ✅ **v1.2 PR Review Cleanup & Test Strategy** — Phases 1-3 (shipped 2026-04-10)
- 🚧 **v1.3 Narwhals Backend for PySpark** — Phases 1-3 (in progress)

## Phases

<details>
<summary>✅ v1.0 Narwhals Backend (Phases 1-5) — SHIPPED 2026-03-15</summary>

- [x] Phase 1: Foundation (2/2 plans) — completed 2026-03-09
- [x] Phase 2: Check Backend (3/3 plans) — completed 2026-03-10
- [x] Phase 3: Column Backend (2/2 plans) — completed 2026-03-14
- [x] Phase 4: Container Backend and Polars Registration (5/5 plans) — completed 2026-03-14
- [x] Phase 5: Ibis Registration and Integration (6/6 plans) — completed 2026-03-15

See `.planning/milestones/v1.0-ROADMAP.md` for full phase details.

</details>

<details>
<summary>✅ v1.1 Ibis Parity & Lazy-First Architecture (Phases 1-9) — SHIPPED 2026-03-25</summary>

- [x] Phase 1: PR Review Architecture Fixes (3/3 plans) — completed 2026-03-22
- [x] Phase 2: Remaining PR Review Fixes (2/2 plans) — completed 2026-03-22
- [x] Phase 3: Fix IbisCheckBackend delegation via apply() type-dispatch (2/2 plans) — completed 2026-03-22
- [x] Phase 4: Lazy postprocess — always-lazy failure_cases (3/3 plans) — completed 2026-03-23
- [x] Phase 5: Expression-based check protocol (3/3 plans) — completed 2026-03-23
- [x] Phase 6: Eliminate unnecessary materialization (3/3 plans) — completed 2026-03-23
- [x] Phase 7: v1.0 Tech Debt Cleanup (2/2 plans) — completed 2026-03-24
- [x] Phase 8: Fix lazy=True critical regressions (2/2 plans) — completed 2026-03-25
- [x] Phase 9: Accumulate check outputs for drop_invalid_rows (2/2 plans) — completed 2026-03-25

See `.planning/milestones/v1.1-ROADMAP.md` for full phase details.

</details>

<details>
<summary>✅ v1.2 PR Review Cleanup & Test Strategy (Phases 1-3) — SHIPPED 2026-04-10</summary>

- [x] Phase 1: Structural Cleanup (6/6 plans) — completed 2026-04-10
- [x] Phase 2: Documentation Polish (2/2 plans) — completed 2026-04-10
- [x] Phase 3: CI Test Strategy (3/3 plans) — completed 2026-04-10

See `.planning/milestones/v1.2-ROADMAP.md` for full phase details.

</details>

### 🚧 v1.3 Narwhals Backend for PySpark (In Progress)

**Milestone Goal:** Wire PySpark into the Narwhals backend via registration, add CI coverage, and document SQL-lazy limitations — making PySpark a first-class supported backend alongside Ibis.

- [ ] **Phase 1: PySpark Registration** — Conditionally wire Narwhals backends for PySpark DataFrames in `register_pyspark_backends()`
- [ ] **Phase 2: Test Coverage and CI** — Run PySpark test suite under narwhals backend, triage failures, add nox session
- [ ] **Phase 3: Documentation** — List PySpark as supported SQL-lazy backend with known limitations

## Phase Details

### Phase 1: PySpark Registration
**Goal**: Users can activate the Narwhals backend for PySpark by setting `PANDERA_USE_NARWHALS_BACKEND=True`, with existing native PySpark behavior unchanged when the flag is off
**Depends on**: Nothing (first phase of v1.3)
**Requirements**: REG-01
**Success Criteria** (what must be TRUE):
  1. Setting `PANDERA_USE_NARWHALS_BACKEND=True` causes `register_pyspark_backends()` to register `NarwhalsCheckBackend`, `ColumnBackend`, and `DataFrameSchemaBackend` for `pyspark_sql.DataFrame`
  2. When `pyspark_connect` is importable, its `DataFrame` type is also registered under the Narwhals backend
  3. Setting `PANDERA_USE_NARWHALS_BACKEND=False` (or leaving it unset) leaves existing native PySpark registrations in place unchanged
  4. The registration wiring follows the same conditional pattern as `register_polars_backends()` and `register_ibis_backends()` — no novel activation mechanism introduced
**Plans**: 1 plan
- [ ] 01-01-PLAN.md — Wire conditional narwhals/native branches into register_pyspark_backends() and add 4 activation/fallback/connect/idempotency tests

### Phase 2: Test Coverage and CI
**Goal**: The existing PySpark test suite runs cleanly under the Narwhals backend, with expected SQL-lazy limitations marked `xfail`, unexpected bugs fixed, and a CI nox session added
**Depends on**: Phase 1
**Requirements**: TEST-01, TEST-02, TEST-03, CI-01
**Success Criteria** (what must be TRUE):
  1. Running `PANDERA_USE_NARWHALS_BACKEND=True pytest tests/pyspark/` produces no unexpected failures — every failure is either a passing test or an `xfail` with a justifying comment
  2. Element-wise checks, `sample=`/`tail=` params, and row-index in `failure_cases` are each covered by at least one `xfail`-marked test documenting the SQL-lazy limitation
  3. Any test failure that is not an expected SQL-lazy limitation (i.e., a true narwhals backend bug) is diagnosed and fixed before this phase closes
  4. A nox session (or parametrized entry) runs `tests/pyspark/` under `PANDERA_USE_NARWHALS_BACKEND=True` with pyspark and narwhals dependencies installed, and that session is listed in the CI matrix
**Plans**: TBD
**UI hint**: no

### Phase 3: Documentation
**Goal**: The narwhals backend documentation clearly lists PySpark as a supported SQL-lazy backend alongside Ibis, with the same limitation notes users see for Ibis
**Depends on**: Phase 1
**Requirements**: DOCS-01
**Success Criteria** (what must be TRUE):
  1. The narwhals backend documentation page names PySpark as a supported SQL-lazy backend (alongside Ibis/DuckDB)
  2. The documentation lists the same SQL-lazy limitations for PySpark that it lists for Ibis: no element-wise checks, no row sampling
  3. A user reading only the narwhals backend docs can determine how to enable PySpark support and what constraints apply, without consulting source code
**Plans**: TBD

## Progress

**Execution Order:** 1 → 2 → 3 (Phase 3 can begin as soon as Phase 1 is complete; Phase 2 requires Phase 1)

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. PySpark Registration | 0/1 | Not started | - |
| 2. Test Coverage and CI | 0/TBD | Not started | - |
| 3. Documentation | 0/TBD | Not started | - |
