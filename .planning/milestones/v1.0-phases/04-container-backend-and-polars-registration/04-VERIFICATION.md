---
phase: 04-container-backend-and-polars-registration
verified: 2026-03-14T14:30:00Z
status: passed
score: 9/9 must-haves verified
re_verification:
  previous_status: gaps_found
  previous_score: 7/9
  gaps_closed:
    - "register_narwhals_backends() wires narwhals as the active Polars backend — including built-in check dispatch"
    - "validate(lazy=True) collects all errors before raising SchemaErrors (does not stop at the first failure)"
  gaps_remaining: []
  regressions: []
---

# Phase 4: Container Backend and Polars Registration — Verification Report

**Phase Goal:** Per-container validation (strict modes, lazy error collection, register_polars_backends auto-detection) works end-to-end with narwhals for Polars frames.
**Verified:** 2026-03-14
**Status:** passed
**Re-verification:** Yes — after gap closure (Plan 04-05)

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | NarwhalsSchemaBackend has failure_cases_metadata returning FailureCaseMetadata with native pl.DataFrame | VERIFIED | `pandera/backends/narwhals/base.py` lines 139-269; test_failure_cases_metadata XPASS |
| 2 | NarwhalsSchemaBackend.drop_invalid_rows() filters rows based on check_output masks | VERIFIED | `pandera/backends/narwhals/base.py` lines 271-295; test_drop_invalid_rows XPASS |
| 3 | PanderaConfig has no use_narwhals_backend field — config.py is clean | VERIFIED | `pandera/config.py` has no mention of use_narwhals_backend; PanderaConfig has 4 fields only |
| 4 | DataFrameSchemaBackend exists with full validate() pipeline — type-preserving | VERIFIED | `pandera/backends/narwhals/container.py` — class exists; validate(), strict_filter_columns(), etc. all present; test_validate_polars_dataframe and test_validate_polars_lazyframe PASSED |
| 5 | strict=True raises SchemaError for extra columns; strict="filter" drops them | VERIFIED | strict_filter_columns() in container.py; test_strict_true_rejects_extra_columns and test_strict_filter_drops_extra_columns PASSED |
| 6 | validate(lazy=True) collects all errors before raising SchemaErrors, each with reason_code=DATAFRAME_CHECK | VERIFIED | test_lazy_mode_collects_all_errors PASSED; each error.reason_code asserted == DATAFRAME_CHECK; production smoke test confirms |
| 7 | SchemaError.failure_cases is a native pl.DataFrame (positive isinstance assertion) | VERIFIED | test_failure_cases_is_native PASSED with `assert isinstance(fc, pl.DataFrame)`; production smoke test confirms |
| 8 | register_polars_backends() auto-detects narwhals via try/except, imports builtin_checks, emits UserWarning | VERIFIED | `pandera/backends/polars/register.py` lines 27-47; UserWarning emitted; builtin_checks imported; test_narwhals_auto_activated_when_installed PASSED |
| 9 | register_polars_backends() is idempotent via lru_cache | VERIFIED | `@lru_cache` on line 9 of polars/register.py; test_register_is_idempotent PASSED |

**Score:** 9/9 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `pandera/backends/polars/register.py` | Auto-detects narwhals via try/except; imports builtin_checks; emits UserWarning | VERIFIED | 57 lines; try block imports narwhals, builtin_checks, NarwhalsCheckBackend, ColumnBackend, DataFrameSchemaBackend; except block falls back to polars backends |
| `pandera/backends/narwhals/register.py` | Stub only — no functions | VERIFIED | 6 lines; docstring only; `def register_narwhals_backends` absent |
| `pandera/backends/narwhals/container.py` | validate() without self-registration block | VERIFIED | No reference to register_narwhals_backends; get_config_context still imported and used for ValidationDepth |
| `pandera/config.py` | PanderaConfig without use_narwhals_backend field | VERIFIED | 4 fields: validation_enabled, validation_depth, cache_dataframe, keep_cached_dataframe; use_narwhals_backend absent throughout |
| `tests/backends/narwhals/conftest.py` | Calls register_polars_backends() in autouse fixture; suppresses UserWarning | VERIFIED | `_suppress_narwhals_warning` fixture calls register_polars_backends() with cache_clear(); suppresses UserWarning |
| `tests/backends/narwhals/test_container.py` | No register_narwhals_backends calls; strengthened assertions | VERIFIED | No register_narwhals_backends anywhere; test_lazy_mode_collects_all_errors asserts reason_code==DATAFRAME_CHECK per error; test_failure_cases_is_native asserts isinstance(fc, pl.DataFrame) |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `pandera/backends/polars/register.py` | `pandera/backends/narwhals/builtin_checks.py` | `from pandera.backends.narwhals import builtin_checks` side-effect inside try block | VERIFIED | Line 30: `from pandera.backends.narwhals import builtin_checks  # noqa — triggers Dispatcher registration for NarwhalsData` |
| `pandera/backends/polars/register.py` | `pandera/backends/narwhals/container.py` | BACKEND_REGISTRY writes for pl.DataFrame and pl.LazyFrame | VERIFIED | Lines 42-43: `DataFrameSchema.register_backend(pl.LazyFrame, DataFrameSchemaBackend)` and `register_backend(pl.DataFrame, DataFrameSchemaBackend)` |
| `pandera/backends/narwhals/container.py` | `pandera/backends/narwhals/base.py` | DataFrameSchemaBackend inherits NarwhalsSchemaBackend | VERIFIED | Line 44: `class DataFrameSchemaBackend(NarwhalsSchemaBackend)` |
| `pandera/backends/narwhals/container.py` | `pandera/backends/narwhals/components.py` | run_schema_component_checks() delegates to ColumnBackend | VERIFIED | validate() delegates column checks via schema_component.validate() calls |
| `tests/backends/narwhals/conftest.py` | `pandera/backends/polars/register.py` | autouse fixture calls register_polars_backends() | VERIFIED | conftest lines 23-25; cache_clear() + register_polars_backends() called per module |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| CONTAINER-01 | 04-01, 04-02 | NarwhalsSchemaBackend has failure_cases_metadata, drop_invalid_rows | SATISFIED | Both methods in base.py; both tests XPASS |
| CONTAINER-02 | 04-01, 04-03, 04-05 | schema.validate(pl.DataFrame/pl.LazyFrame) works end-to-end — production and test | SATISFIED | Production smoke test: SchemaError with reason_code=DATAFRAME_CHECK, failure_cases is pl.DataFrame; all validate tests PASSED |
| CONTAINER-03 | 04-01, 04-03 | strict=True raises SchemaError; strict="filter" drops extra columns | SATISFIED | strict_filter_columns() correct; both tests PASSED |
| CONTAINER-04 | 04-01, 04-03, 04-05 | lazy=True collects all errors; each has reason_code=DATAFRAME_CHECK | SATISFIED | test_lazy_mode_collects_all_errors PASSED with strengthened per-error reason_code assertion |
| REGISTER-01 | 04-01, 04-04 | register_polars_backends() is idempotent via lru_cache | SATISFIED | @lru_cache on register_polars_backends(); test_register_is_idempotent PASSED |
| REGISTER-02 | 04-01, 04-04 | Narwhals backend registers for pl.DataFrame and pl.LazyFrame | SATISFIED | BACKEND_REGISTRY writes in polars/register.py lines 42-47; test_polars_backends_registered PASSED |
| REGISTER-04 | 04-01, 04-05 | Narwhals backend auto-activated when narwhals is installed | SATISFIED | register_polars_backends() auto-detects narwhals via try/except; UserWarning emitted; test_narwhals_auto_activated_when_installed PASSED |
| TEST-03 | 04-01, 04-03, 04-05 | SchemaError.failure_cases is always a native frame type | SATISFIED | test_failure_cases_is_native PASSED with positive isinstance(fc, pl.DataFrame) assertion; production smoke test confirms |

**Orphaned requirements:** None. All 8 requirement IDs in plans are accounted for in REQUIREMENTS.md Phase 4.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `pandera/backends/narwhals/base.py` | 178 | `raise NotImplementedError` for pl.LazyFrame failure_cases | Info | Expected limitation: documented design decision for Phase 4 |
| `pandera/backends/narwhals/base.py` | 55 | `raise NotImplementedError` for sample= | Info | Expected: sample-based subsampling deferred; documented in docstring |
| `tests/backends/narwhals/test_container.py` | 28, 48 | `xfail` markers on test_failure_cases_metadata and test_drop_invalid_rows | Warning | Both tests are XPASS (implementations exist and work); xfail markers should be removed in a follow-up. Not a blocker — tests pass. |

---

### Human Verification Required

None. All gap closure items verified programmatically:

- Production smoke test (no conftest, fresh process): `SchemaError.reason_code == DATAFRAME_CHECK` and `isinstance(failure_cases, pl.DataFrame)` — confirmed.
- UserWarning emitted test (`-W error::UserWarning`): UserWarning with 'Narwhals' in message — confirmed.
- Full test suite: 10 passed, 2 xpassed, 0 failed — confirmed.

---

### Re-verification Summary

**Previous gaps (both closed):**

1. **register_narwhals_backends() missing builtin_checks import** — CLOSED. The architectural decision changed: `register_polars_backends()` now auto-detects narwhals and imports `pandera.backends.narwhals.builtin_checks` as a side-effect inside its try block (line 30). The separate `register_narwhals_backends()` function was eliminated entirely. Production schema.validate() with builtin checks now raises `SchemaError` with `reason_code=DATAFRAME_CHECK` and native `pl.DataFrame` failure_cases.

2. **lazy=True errors had wrong reason_code (CHECK_ERROR instead of DATAFRAME_CHECK)** — CLOSED. Root cause was the same missing builtin_checks import; fixed by the architectural change above. Additionally, `test_lazy_mode_collects_all_errors` was strengthened to assert `err.reason_code == SchemaErrorReason.DATAFRAME_CHECK` for each error — this assertion now passes.

**No regressions detected.** All truths previously verified (1, 2, 3, 4, 5, 8, 9) still pass. The two partial truths (6, 7) now fully pass.

---

_Verified: 2026-03-14_
_Verifier: Claude (gsd-verifier)_
