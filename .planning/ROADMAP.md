# Roadmap: Pandera Narwhals Backend

## Milestones

- ✅ **v1.0 Narwhals Backend** — Phases 1-5 (shipped 2026-03-15)
- ✅ **v1.1 Ibis Parity & Lazy-First Architecture** — Phases 1-9 (shipped 2026-03-25)
- ✅ **v1.2 PR Review Cleanup & Test Strategy** — Phases 1-3 (shipped 2026-04-10)
- 🔄 **v1.3 Narwhals Backend for PySpark** — Phases 1-6 (in progress — PR #2339 open)

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

### 🔄 v1.3 Narwhals Backend for PySpark (In Progress — PR #2339)

**Milestone Goal:** Wire PySpark into the Narwhals backend via registration, add CI coverage, document SQL-lazy limitations, and address all pre-merge review findings before the PR ships.

- [x] **Phase 1: PySpark Registration** — Conditionally wire Narwhals backends for PySpark DataFrames in `register_pyspark_backends()` — completed 2026-05-10 (1/1 plans)
- [x] **Phase 2: Test Coverage and CI** — Run PySpark test suite under narwhals backend, triage failures, add nox session — completed 2026-05-18 (4/4 plans)
- [x] **Phase 3: Documentation** — List PySpark as supported SQL-lazy backend with known limitations — completed 2026-05-18 (1/1 plans)
- [x] **Phase 4: Eliminate Backend-Specific Dispatch Branches** — Remove or fix the four `is_pyspark` dispatch violations in `base.py`, `components.py`, and `container.py`; fix `_concat_failure_cases` silent-drop bug (0/4 plans) (completed 2026-05-25)
- [ ] **Phase 5: Correctness and Behavioral Parity** — Fix `strict='filter'` no-op, add `pandera.schema` after narwhals validation, fix `test_pyspark_config.py` band-aid xfails (0/2 plans)
- [ ] **Phase 6: Test Coverage and Minor Fixes** — Add PySpark to `tests/narwhals/test_e2e.py`, fix CI Python version comment, fix "not in dataframe" message, fix registration test, fix stacked xfails, fix `supported_types()` double-append (0/0 plans)

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

- [x] 01-01-PLAN.md — Wire conditional narwhals/native branches into register_pyspark_backends() and add 4 activation/fallback/connect/idempotency tests — complete 2026-05-10

### Phase 2: Test Coverage and CI

**Goal**: The existing PySpark test suite runs cleanly under the Narwhals backend, with expected SQL-lazy limitations marked `xfail`, unexpected bugs fixed, and a CI nox session added
**Depends on**: Phase 1
**Requirements**: TEST-01, TEST-02, TEST-03, CI-01
**Success Criteria** (what must be TRUE):

  1. Running `PANDERA_USE_NARWHALS_BACKEND=True pytest tests/pyspark/` produces no unexpected failures — every failure is either a passing test or an `xfail` with a justifying comment
  2. Element-wise checks and `sample=`/`tail=` params are each covered by at least one `xfail`-marked test documenting the SQL-lazy limitation. Row-index in `failure_cases` is inapplicable to PySpark — PySpark DataFrames are distributed and partitioned with no native integer row index; this clause applies only to polars/ibis backends (see 02-03-TRIAGE.md ## SC2c Decision)
  3. Any test failure that is not an expected SQL-lazy limitation (i.e., a true narwhals backend bug) is diagnosed and fixed before this phase closes
  4. A nox session (or parametrized entry) runs `tests/pyspark/` under `PANDERA_USE_NARWHALS_BACKEND=True` with pyspark and narwhals dependencies installed, and that session is listed in the CI matrix

**Plans**: 4 plans

- [x] 02-01-PLAN.md — Apply xfail markers to 11 expected SQL-lazy limitation tests across 4 pyspark test files (TEST-02)
- [x] 02-02-PLAN.md — Extend tests_narwhals_backend nox session and CI matrix to cover pyspark extra (CI-01)
- [x] 02-03-PLAN.md — Triage tests/pyspark/ under narwhals: produce TRIAGE.md, apply any additional xfails, fix true backend bugs (TEST-01, TEST-03)
- [x] 02-04-PLAN.md — Gap-closure: fix 4 Category C narwhals backend bugs from TRIAGE.md (SchemaErrors→pandera.errors, _concat_failure_cases PySpark dispatch, dtype string format, materialization avoidance) and resolve SC2c row-index applicability (TEST-01, TEST-03) — complete 2026-05-18

**UI hint**: no

### Phase 3: Documentation

**Goal**: The narwhals backend documentation clearly lists PySpark as a supported SQL-lazy backend alongside Ibis, with the same limitation notes users see for Ibis
**Depends on**: Phase 1
**Requirements**: DOCS-01
**Success Criteria** (what must be TRUE):

  1. The narwhals backend documentation page names PySpark as a supported SQL-lazy backend (alongside Ibis/DuckDB)
  2. The documentation lists the same SQL-lazy limitations for PySpark that it lists for Ibis: no element-wise checks, no row sampling
  3. A user reading only the narwhals backend docs can determine how to enable PySpark support and what constraints apply, without consulting source code

**Plans**: 1 plan

- [x] 03-01-PLAN.md — Add narwhals opt-in note to pyspark_sql.md and add PySpark to the narwhals-backends content in supported_libraries.md (DOCS-01) — complete 2026-05-18

### Phase 4: Eliminate Backend-Specific Dispatch Branches

**Goal**: The narwhals backend has no `is_pyspark` special-casing — all four dispatch violations from PR review are removed or properly abstracted
**Depends on**: Phase 1
**Requirements**: ARCH-01, ARCH-02, ARCH-03, ARCH-04
**Success Criteria** (what must be TRUE):

  1. The `run_check` PySpark branch in `base.py:143-154` is either removed (if the `_materialize()` comment is stale) or replaced with a concrete justification and failing test; no module-level `implementation in (PYSPARK, PYSPARK_CONNECT)` check remains in `run_check`
  2. `_concat_failure_cases` uses narwhals-native concatenation without module-string sniffing; non-PySpark scalar frames (e.g. `_build_scalar_failure_case` output) are no longer silently dropped when PySpark frames are present
  3. `check_dtype` in `components.py` uses narwhals-native dtype comparison rather than a PySpark-specific `str(pyspark_dtype) == str(schema.dtype)` path; the test workaround in `test_pyspark_dtypes.py` is explained or removed
  4. The PySpark error-setting logic in `container.py:validate()` is extracted to a method rather than an inline `is_pyspark` test

**Plans**: 4 plans

- [x] 04-01-PLAN.md — Fix `_materialize()` in `pandera/api/narwhals/utils.py` to handle PySpark via `.first()` + pyarrow; delete the PySpark branch in `run_check` (ARCH-01)
- [x] 04-02-PLAN.md — Refactor `_build_lazy_failure_case` to return narwhals-wrapped frames; rewrite `_concat_failure_cases` to dispatch on `nw.Implementation` instead of module-string sniffing; surface dropped scalar items via `SchemaWarning` (ARCH-02)
- [x] 04-03-PLAN.md — Replace frame-implementation probe in `check_dtype` with `isinstance(schema.dtype, pyspark_engine.DataType)` schema-driven probe; add explanatory comment to the verifySchema=False workaround in `test_pyspark_dtypes.py` (ARCH-03)
- [x] 04-04-PLAN.md — Extract `_handle_pyspark_validation_result()` instance method on `DataFrameSchemaBackend`; replace the two inline `is_pyspark` blocks in `validate()` with method calls; keep the `is_pyspark` dispatch detection (ARCH-04)

### Phase 5: Correctness and Behavioral Parity

**Goal**: The narwhals PySpark success path has behavioral parity with the native backend — `strict='filter'` applies filtering, `df.pandera.schema` is set, and band-aid xfails in config tests are removed
**Depends on**: Phase 4
**Requirements**: CORR-01, CORR-02, TEST-FIX-01
**Success Criteria** (what must be TRUE):

  1. `strict='filter'` returns filtered columns for PySpark narwhals in the success path — `_to_frame_kind_nw(check_lf, return_type)` is returned with `errors = {}` attached; the corresponding xfail in `test_pyspark_model.py` is removed or converted to a passing test
  2. `check_obj.pandera.add_schema(schema)` is called before returning from narwhals PySpark validation; the xfail in `test_pyspark_accessor.py` is removed or converted to a passing test
  3. The five `test_pyspark_config.py` tests that xfail due to hardcoded `"use_narwhals_backend": False` in expected dicts are fixed — either updated to use `CONFIG.use_narwhals_backend` dynamically or the key is removed from the assertion**Plans**: 2 plans
- [ ] 05-01-PLAN.md — Fix CORR-01 + CORR-02 in narwhals container.py (pass `_to_frame_kind_nw(check_lf, return_type)` at both PySpark return sites; call `check_obj.pandera.add_schema(schema)` in `_handle_pyspark_validation_result`), extend ARCH-04 unit tests with `add_schema` assertions, and remove the two now-passing xfails in `test_pyspark_model.py` and `test_pyspark_accessor.py` (CORR-01, CORR-02)
- [ ] 05-02-PLAN.md — Remove 5 band-aid xfail decorators on `TestPanderaConfig` in `test_pyspark_config.py` and replace hardcoded `"use_narwhals_backend": False` with `CONFIG.use_narwhals_backend` in the five `expected` dicts (TEST-FIX-01)

**UI hint**: no

### Phase 6: Test Coverage and Minor Fixes

**Goal**: PySpark narwhals coverage exists in `tests/narwhals/test_e2e.py` and all minor issues from the PR review are resolved
**Depends on**: Phase 5
**Requirements**: TEST-E2E-01, NITS-01
**Success Criteria** (what must be TRUE):

  1. `tests/narwhals/test_e2e.py` includes a PySpark section with at minimum: backend registration assertion, return-type preservation (validate → returns original PySpark DataFrame), a passing and failing built-in check with failure cases inspected, and nullable/unique check behavior
  2. The CI Python version exclusion for PySpark (3.12, 3.13) has a comment explaining PySpark's maximum supported Python version
  3. `container.py` emits a backend-neutral "column 'X' not found" message instead of hardcoding "not in dataframe" — the corresponding xfail in `test_ibis_container.py` is removed
  4. `test_pyspark_narwhals_register.py::test_pyspark_narwhals_activated_when_opted_in` asserts all three backends (`NarwhalsCheckBackend`, `ColumnBackend`, `DataFrameSchemaBackend`) are registered
  5. The stacked `@pytest.mark.xfail` decorators in `test_pyspark_model.py::test_registered_dataframemodel_checks` are combined into a single conditional xfail
  6. The `supported_types()` double-append of `PySparkSQLDataFrame` in `pandera/api/pyspark/types.py` is fixed

**Plans**: 0 plans

## Progress

**Execution Order:** 1 → 2 → 3 (Phase 3 can start after Phase 1); 4 → 5 → 6

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. PySpark Registration | 1/1 | Complete ✓ | 2026-05-10 |
| 2. Test Coverage and CI | 4/4 | Complete ✓ | 2026-05-18 |
| 3. Documentation | 1/1 | Complete ✓ | 2026-05-18 |
| 4. Eliminate Backend-Specific Dispatch Branches | 4/4 | Complete    | 2026-05-25 |
| 5. Correctness and Behavioral Parity | 0/2 | Planned | — |
| 6. Test Coverage and Minor Fixes | 0/TBD | Planned | — |

## Backlog

### Phase 999.3: Define PySparkData wrapper for native=True checks (BACKLOG)

**Goal:** Provide a typed `PySparkData` wrapper analogous to `PolarsData`/`IbisData` for users writing `native=True` checks under the narwhals backend for PySpark.
**Context:** `_wrap_native_frame_with_key()` in `checks.py` returns `None` for PySpark frames, causing `native=True` checks to receive a raw `pyspark.sql.DataFrame` via the 2-arg legacy fallback. This is undocumented and inconsistent with Polars/Ibis convention. Requires defining a `PySparkData` NamedTuple and wiring it through `_wrap_native_frame_with_key`.
**See:** `pandera/backends/narwhals/checks.py:_wrap_native_frame_with_key`, `pandera/typing/pyspark_sql.py`
**Plans:** 0 plans

Plans:

- [ ] TBD (promote with /gsd-review-backlog when ready)

### Phase 999.1: Relax PySpark native backend type restrictions (BACKLOG)

**Goal:** Captured for future planning
**Requirements:** TBD
**Context:** The native PySpark backend's `@register_input_datatypes` decorator restricts `ge`/`gt`/`lt`/`le` to numeric+date types and `isin`/`notin` to numeric+date+string+binary — but PySpark SQL natively supports string and boolean comparisons, and Ibis imposes no such restrictions. The narwhals backend is already more permissive (aligned with Ibis), which is why several `test_failed_unaccepted_datatypes` tests need xfail under narwhals. Relaxing these restrictions in the native PySpark backend would bring it into alignment with Ibis and narwhals, and eliminate those xfails.
**See:** `pandera/backends/pyspark/builtin_checks.py`, `pandera/backends/pyspark/decorators.py:register_input_datatypes`
**Plans:** 0 plans

Plans:

- [ ] TBD (promote with /gsd-review-backlog when ready)

### Phase 999.2: Test and verify narwhals ColumnBackend regex support across all backends (BACKLOG)

**Goal:** Add explicit test coverage for `regex=True` columns under `use_narwhals_backend=True` for polars and ibis; confirm no gaps exist.
**Context:** The narwhals `ColumnBackend` (`pandera/backends/narwhals/components.py`) is the shared column backend used by polars, ibis, and PySpark when `use_narwhals_backend=True`. Regex expansion logic was added during v1.3 PySpark work (missing from the original narwhals ColumnBackend). Polars/ibis regex tests only run against the native per-backend ColumnBackends (`use_narwhals_backend=False`), so the narwhals path has no regex test coverage for those backends. A follow-up PR should add those tests and fix any gaps found.
**See:** `pandera/backends/narwhals/components.py`, `tests/polars/test_polars_components.py::test_column_schema_regex`
**Plans:** 0 plans

Plans:

- [ ] TBD (promote with /gsd-review-backlog when ready)
