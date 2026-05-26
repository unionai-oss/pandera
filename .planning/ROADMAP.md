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
- [x] **Phase 5: Correctness and Behavioral Parity** — Fix `strict='filter'` no-op, add `pandera.schema` after narwhals validation, fix `test_pyspark_config.py` band-aid xfails (0/2 plans) (completed 2026-05-25)
- [x] **Phase 6: Test Coverage and Minor Fixes** — Add PySpark to `tests/narwhals/test_e2e.py`, fix CI Python version comment, fix "not in dataframe" message, fix registration test, fix stacked xfails, fix `supported_types()` double-append (0/2 plans) (completed 2026-05-25)

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
- [x] 05-01-PLAN.md — Fix CORR-01 + CORR-02 in narwhals container.py (pass `_to_frame_kind_nw(check_lf, return_type)` at both PySpark return sites; call `check_obj.pandera.add_schema(schema)` in `_handle_pyspark_validation_result`), extend ARCH-04 unit tests with `add_schema` assertions, and remove the two now-passing xfails in `test_pyspark_model.py` and `test_pyspark_accessor.py` (CORR-01, CORR-02)
- [x] 05-02-PLAN.md — Remove 5 band-aid xfail decorators on `TestPanderaConfig` in `test_pyspark_config.py` and replace hardcoded `"use_narwhals_backend": False` with `CONFIG.use_narwhals_backend` in the five `expected` dicts (TEST-FIX-01)

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

**Plans**: 2 plans

- [x] 06-01-PLAN.md — Add PySpark section to tests/narwhals/test_e2e.py (registration, return-type, passing+failing check, nullable, unique) (TEST-E2E-01)
- [x] 06-02-PLAN.md — Resolve five pre-merge nits: CI Python comment, backend-neutral column error message, expanded registration test, collapsed stacked xfail, supported_types() double-append (NITS-01)

## Progress

**Execution Order:** 1 → 2 → 3 (Phase 3 can start after Phase 1); 4 → 5 → 6 → 7 → 8

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. PySpark Registration | 1/1 | Complete ✓ | 2026-05-10 |
| 2. Test Coverage and CI | 4/4 | Complete ✓ | 2026-05-18 |
| 3. Documentation | 1/1 | Complete ✓ | 2026-05-18 |
| 4. Eliminate Backend-Specific Dispatch Branches | 4/4 | Complete    | 2026-05-25 |
| 5. Correctness and Behavioral Parity | 2/2 | Complete    | 2026-05-25 |
| 6. Test Coverage and Minor Fixes | 2/2 | Complete    | 2026-05-26 |
| 7. CI Fixes and Post-Review Quick Fixes | 2/2 | Complete    | 2026-05-26 |
| 8. Test Quality Improvements | 3/3 | Complete    | 2026-05-26 |

### Phase 7: CI Fixes and Post-Review Quick Fixes

**Goal:** Fix the two CI failures introduced by Phase 6 and resolve mechanical post-review nits — no architectural decisions, just correct the branch so CI goes green
**Requirements**: CI-FIX-01, CI-FIX-02, NITS-02
**Depends on:** Phase 6
**Success Criteria** (what must be TRUE):

  1. `Unit Tests Narwhals Backend (polars)` passes — `test_column_absent_error` is updated to match the current "not found" message, OR the narwhals container reverts to "not in dataframe" and the polars test requires no change (decision: revert to "not in dataframe" since pandera is a dataframe library and the message is not wrong)
  2. `Unit Tests Narwhals` passes — the `_spark_env_vars` autouse fixture in `tests/narwhals/test_e2e.py` yields instead of returning on the `HAS_PYSPARK=False` early-exit path, so pytest does not raise `ValueError: fixture did not yield a value`
  3. The redundant `assert native_pyspark_schema is not None` in `pandera/backends/narwhals/components.py::check_dtype` is removed — the surrounding `if uses_pyspark_dtype:` guard makes it structurally unreachable; use a proper `if`/`raise` or simply trust the guard
  4. The `tests/common/` exclusion for PySpark in the noxfile `tests_narwhals_backend` session has an inline comment explaining why (e.g. no `pyspark` marker exists in `tests/common/`)
  5. "Pyspark SQL" occurrences in `docs/source/supported_libraries.md` are corrected to "PySpark SQL"

**Plans:** 2/2 plans complete
Plans:

- [x] 07-01-PLAN.md — Revert narwhals container COLUMN_NOT_IN_DATAFRAME message to "not in dataframe" + restore ibis test xfail + fix _spark_env_vars yield (CI-FIX-01, CI-FIX-02)
- [x] 07-02-PLAN.md — Remove redundant non-None assert in check_dtype + add noxfile tests/common/ exclusion comment + fix Pyspark SQL casing in supported_libraries.md (NITS-02)

### Phase 8: Test Quality Improvements

**Goal:** Replace test anti-patterns identified in the updated PR review with idiomatic, maintainable alternatives — no production code changes except the `_concat_failure_cases` polars-branch fix, only test improvements
**Requirements**: TQ-01, TQ-02, TQ-03, TQ-04
**Depends on:** Phase 7
**Success Criteria** (what must be TRUE):

  1. `tests/pyspark/test_pyspark_error.py` no longer contains inline `if CONFIG.use_narwhals_backend else` ternaries for expected error strings — the six assertions are updated to use the `_cmp_errors` helper pattern from `test_pyspark_config.py` (or an equivalent that asserts structural fields and only that `error` is non-empty)
  2. `_concat_failure_cases` in `pandera/backends/narwhals/base.py` — the polars branch either (a) gains a comment proving that `pl_items` is always empty for polars narwhals validation (so the silent drop is safe), or (b) is fixed to concatenate `pl_items` as well, matching the PySpark branch which at least warns
  3. `tests/narwhals/test_arch03_schema_driven_dispatch.py` source-inspection tests that assert the presence or absence of specific variable names inside method bodies are replaced with behavioral equivalents — call `check_dtype` directly with a schema and frame combination that exercises the relevant code path
  4. The PySpark narwhals registration either registers `Check.register_backend(nw.DataFrame, NarwhalsCheckBackend)` to match the polars narwhals path, or adds a comment in `register_pyspark_backends()` explaining why `nw.DataFrame` is not needed for PySpark

**Plans:** 3/3 plans complete
Plans:

- [x] 08-01-PLAN.md — Extract `_cmp_errors` to `tests/pyspark/conftest.py` as a module-level function; convert `TestPanderaConfig._cmp_errors` to a delegation call; replace 6 CONFIG ternaries in `test_pyspark_error.py` DATA assertions with `_cmp_errors`; drop redundant `error` keys; remove unused CONFIG import (TQ-01)
- [x] 08-02-PLAN.md — Add regression test for `_concat_failure_cases` polars branch (RED), then fix the polars branch in `pandera/backends/narwhals/base.py` to merge `pl_items` via `pl.concat([lazy_result.collect()] + pl_items)` without warning (TQ-02)
- [x] 08-03-PLAN.md — Delete 4 source-inspection tests in `tests/narwhals/test_arch03_schema_driven_dispatch.py`; keep the 5th behavioral test; add 2 PySpark-gated behavioral tests for the PySpark-dtype dispatch path; add a comment in `pandera/backends/pyspark/register.py` documenting the intentional `nw.DataFrame` omission (consistent with ibis precedent) (TQ-03, TQ-04)

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
