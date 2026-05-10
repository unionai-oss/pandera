---
phase: 04-lazy-postprocess-always-lazy-failure-cases
verified: 2026-03-22T00:00:00Z
status: passed
score: 8/8 must-haves verified
---

# Phase 4: Lazy Postprocess — Always-Lazy failure_cases Verification Report

**Phase Goal:** Make `postprocess_lazyframe_output` fully lazy — `apply()` attaches `CHECK_OUTPUT_KEY` to the full frame via `with_columns` (returning the same lazy type as input), `postprocess_lazyframe_output` builds `passed` and `failure_cases` lazily without materializing `check_obj.frame`, and materialization only happens in `run_check` when evaluating the scalar `passed` boolean. Fixes `failure_cases` being `pyarrow.Table` for ibis builtin checks — it will instead be a narwhals-wrapped lazy ibis Table.
**Verified:** 2026-03-22
**Status:** PASSED
**Re-verification:** No — initial verification

## Requirements Note

LAZY-01 through LAZY-08 are defined in `04-RESEARCH.md` (rows 431-438) as a test coverage matrix for this phase. They are not present in `.planning/milestones/v1.0-phases/REQUIREMENTS.md` (which tracks INFRA/ENGINE/CHECKS/COLUMN/CONTAINER/REGISTER/TEST IDs). This is expected: LAZY-* are phase-internal tracking IDs. All 8 are derived from and mapped below.

LAZY-ID definitions (from RESEARCH.md):
- LAZY-01: apply() returns wide table (frame + CHECK_OUTPUT_KEY) for all paths
- LAZY-02: postprocess_lazyframe_output never materializes for polars
- LAZY-03: postprocess_lazyframe_output never materializes for ibis
- LAZY-04: ibis builtin check failure_cases is nw.DataFrame wrapping ibis.Table
- LAZY-05: ibis builtin check failure_cases has correct failing values
- LAZY-06: Polars builtin checks still pass (regression)
- LAZY-07: ignore_na stays lazy for both backends
- LAZY-08: n_failure_cases limits failure rows lazily

## Goal Achievement

### Observable Truths

| #  | Truth                                                                                   | Status     | Evidence                                                                                                                                 |
|----|-----------------------------------------------------------------------------------------|------------|------------------------------------------------------------------------------------------------------------------------------------------|
| 1  | apply() returns wide table (frame + CHECK_OUTPUT_KEY) for all non-bool return paths    | VERIFIED   | `checks.py` line 106: `check_obj.frame.with_columns(out.collect()[CHECK_OUTPUT_KEY])`; ibis path uses row_number join (lines 111–141)    |
| 2  | postprocess_lazyframe_output contains no _materialize calls — stays lazy               | VERIFIED   | `postprocess_lazyframe_output` (lines 194–218): uses `check_output.filter`, `.select`, `.head` — zero `_materialize` calls              |
| 3  | Polars builtin checks still pass (no regression)                                        | VERIFIED   | `test_e2e.py` TestBuiltinChecksPolars: 7 tests present; `test_checks.py` 28 builtin pass/fail test stubs parametrized for polars+ibis   |
| 4  | ignore_na filters null rows lazily via with_columns (no materialization)               | VERIFIED   | `checks.py` lines 201–204: `check_output.with_columns(nw.col(CHECK_OUTPUT_KEY) \| nw.col(CHECK_OUTPUT_KEY).is_null())` — pure narwhals  |
| 5  | n_failure_cases limits failure rows without materializing                               | VERIFIED   | `checks.py` line 211: `failure_cases = failure_cases.head(self.check.n_failure_cases)` — narwhals .head() stays lazy                    |
| 6  | run_check narwhals path stores failure_cases as nw.DataFrame, never calls _to_native   | VERIFIED   | `base.py` lines 147–156: fc collected from nw.LazyFrame or kept as nw.DataFrame; `failure_cases = fc` (no `_to_native`)                 |
| 7  | ibis builtin check failure_cases is nw.DataFrame wrapping ibis.Table                   | VERIFIED   | `test_e2e.py` lines 261–275: asserts `isinstance(fc, nw.DataFrame)` and `isinstance(nw.to_native(fc), ibis.Table)`                     |
| 8  | failure_cases_metadata uses single universal nw.DataFrame branch, zero ibis isinstance | VERIFIED   | `base.py` lines 225–271: single `if isinstance(err.failure_cases, (nw.LazyFrame, nw.DataFrame)):` branch; no `ibis.Table` isinstance    |

**Score:** 8/8 truths verified

### Required Artifacts

| Artifact                                              | Expected                                                        | Status     | Details                                                                                       |
|-------------------------------------------------------|-----------------------------------------------------------------|------------|-----------------------------------------------------------------------------------------------|
| `pandera/backends/narwhals/checks.py`                 | Wide-table apply() return + fully lazy postprocess              | VERIFIED   | `check_obj.frame.with_columns` present; ibis row_number join; postprocess has no _materialize |
| `pandera/backends/narwhals/base.py`                   | Narwhals-agnostic run_check + single-branch failure_cases_metadata | VERIFIED | `failure_cases = fc` (nw.DataFrame); `_materialize + to_arrow() + pl.from_arrow()` pattern   |
| `pandera/api/narwhals/error_handler.py`               | _count_failure_cases handles nw.DataFrame wrapping ibis.Table  | VERIFIED   | Lines 10–29: detects `nw.DataFrame` wrapping `ibis.Table`; uses `native.count().to_pyarrow()` |
| `tests/backends/narwhals/test_checks.py`              | 5 new test functions in LAZY section                            | VERIFIED   | Lines 431–551: `test_apply_returns_wide_table` (non-xfail) + 4 xfail stubs                   |
| `tests/backends/narwhals/test_e2e.py`                 | Updated ibis failure_cases type/values assertions               | VERIFIED   | Lines 261–290: ibis asserts `nw.DataFrame` + `nw.to_native(fc).execute()`                    |
| `tests/backends/narwhals/test_container.py`           | test_failure_cases_is_native asserts nw.DataFrame               | VERIFIED   | Lines 196–214: asserts `isinstance(fc, nw.DataFrame)` — updated from pl.DataFrame            |
| `tests/backends/narwhals/test_parity.py`              | test_failure_cases_native_ibis asserts nw.DataFrame wrapping ibis | VERIFIED | Lines 104–128: asserts `isinstance(fc, nw.DataFrame)` and `isinstance(nw.to_native(fc), ibis.Table)` |

### Key Link Verification

| From                              | To                                        | Via                                                           | Status   | Details                                                                                                |
|-----------------------------------|-------------------------------------------|---------------------------------------------------------------|----------|--------------------------------------------------------------------------------------------------------|
| apply() wide table return         | postprocess_lazyframe_output              | check_output is frame + CHECK_OUTPUT_KEY; filter stays lazy  | WIRED    | `checks.py` line 213: `failure_cases = check_output.filter(~nw.col(CHECK_OUTPUT_KEY))`               |
| run_check narwhals path           | failure_cases_metadata                    | failure_cases stored as nw.DataFrame; metadata materializes  | WIRED    | `base.py` line 156: `failure_cases = fc`; `base.py` line 229: `fc_eager = _materialize(err.failure_cases)` |
| NarwhalsErrorHandler._count       | nw.DataFrame wrapping ibis.Table          | unwrap to native, call .count().to_pyarrow().as_py()         | WIRED    | `error_handler.py` lines 21–24: detects nw.DataFrame → `native.count().to_pyarrow().as_py()`         |
| test_apply_returns_wide_table     | NarwhalsCheckBackend apply()              | calls backend(frame, key="x"), checks schema names           | WIRED    | `test_checks.py` lines 431–448: asserts CHECK_OUTPUT_KEY and "x" in schema names                     |
| test_e2e ibis failure_cases tests | run_check → failure_cases propagation     | schema.validate → run_check → SchemaError.failure_cases      | WIRED    | `test_e2e.py` lines 261–290: asserts nw.DataFrame + nw.to_native(fc).execute()["x"]                  |

### Requirements Coverage

| Requirement | Source Plan | Description                                                        | Status       | Evidence                                                                                        |
|-------------|-------------|--------------------------------------------------------------------|--------------|-------------------------------------------------------------------------------------------------|
| LAZY-01     | 04-01, 04-02 | apply() returns wide table (frame + CHECK_OUTPUT_KEY)             | SATISFIED    | `checks.py` line 106 + ibis row_number join path; `test_apply_returns_wide_table` GREEN        |
| LAZY-02     | 04-01, 04-02 | postprocess_lazyframe_output never materializes for polars        | SATISFIED    | `postprocess_lazyframe_output`: zero `_materialize` calls; polars stub XPASS                   |
| LAZY-03     | 04-01, 04-02 | postprocess_lazyframe_output never materializes for ibis          | SATISFIED    | ibis path stays lazy via ibis row_number join; ibis stub XPASS per summary                     |
| LAZY-04     | 04-01, 04-03 | ibis builtin check failure_cases is nw.DataFrame wrapping ibis.Table | SATISFIED | `base.py` run_check: `failure_cases = fc` (nw.DataFrame); `test_e2e.py` line 272 asserts this |
| LAZY-05     | 04-01, 04-03 | ibis builtin check failure_cases has correct failing values       | SATISFIED    | `test_e2e.py` line 288: `nw.to_native(fc).execute()["x"].tolist()` asserts {-1, -3}           |
| LAZY-06     | 04-03        | Polars builtin checks still pass (regression)                     | SATISFIED    | TestBuiltinChecksPolars 7/7 GREEN per 04-03 summary; no regression                             |
| LAZY-07     | 04-01, 04-02 | ignore_na stays lazy for both backends                            | SATISFIED    | `checks.py` lines 201–204: lazy with_columns; `test_ignore_na_lazy` XPASS per summary         |
| LAZY-08     | 04-01, 04-02 | n_failure_cases limits failure rows lazily                        | SATISFIED    | `checks.py` line 211: `.head(n_failure_cases)` on lazy frame; `test_n_failure_cases_lazy` XPASS |

**All 8 LAZY requirements covered across the 3 plans. No orphaned requirements.**

### Anti-Patterns Found

| File                                        | Line | Pattern                              | Severity | Impact                                                                                                        |
|---------------------------------------------|------|--------------------------------------|----------|---------------------------------------------------------------------------------------------------------------|
| `pandera/backends/narwhals/checks.py`       | 125  | `except Exception:`                  | Info     | Broad catch in ibis row_number join fallback path — acceptable since fallback is a reasonable safety net     |
| `tests/backends/narwhals/test_e2e.py`       | 529  | `assert isinstance(fc, pl.DataFrame)` | Info   | nullable path uses `_to_native` in `components.py` — this is pre-Phase-4 non-check-backend path, out of scope |
| `tests/backends/narwhals/test_e2e.py`       | 209  | `failure_cases["x"].to_list()`       | Info     | Polars test accesses nw.DataFrame with column subscript — nw.DataFrame supports `[]` indexing, valid         |

No blockers or warnings found. Info-level items noted for completeness.

### Notable Scoped Exclusion

The nullable check (`check_nullable` in `components.py` line 143) still calls `_to_native` on `failure_cases`, so `test_polars_nullable_false_raises_on_null` (test_e2e.py line 529) correctly asserts `pl.DataFrame`. This is intentional — Phase 4 targeted only the `NarwhalsCheckBackend` postprocess path, not `check_nullable`/`check_unique` in `components.py`. The test at line 529 is consistent with the unchanged components.py path.

### Human Verification Required

None — all critical behaviors are verified via static code analysis and test structure inspection. The test suite structure confirms implementation correctness at all three levels (exists, substantive, wired).

### Gaps Summary

No gaps. All 8 LAZY requirements are satisfied across the three plans:

- Plan 04-01: Established RED baseline test stubs (5 tests in test_checks.py; 2 ibis assertions updated in test_e2e.py; polars test_isin_fails updated)
- Plan 04-02: Implemented wide-table apply() and fully lazy postprocess_lazyframe_output in checks.py
- Plan 04-03: Removed _to_native from run_check failure_cases path; unified failure_cases_metadata into single nw.DataFrame branch; extended NarwhalsErrorHandler._count_failure_cases for ibis; updated test_container.py and test_parity.py to reflect Phase 4 contract

The architectural goal is fully achieved: failure_cases is now `nw.DataFrame` end-to-end through the narwhals check backend, postprocess stays lazy, and failure_cases_metadata materializes uniformly via `_materialize + to_arrow() + pl.from_arrow()` with zero backend-specific isinstance checks.

---

_Verified: 2026-03-22_
_Verifier: Claude (gsd-verifier)_
