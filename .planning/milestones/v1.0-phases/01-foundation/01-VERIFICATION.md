---
phase: 01-foundation
verified: 2026-03-09T23:30:00Z
status: passed
score: 9/9 must-haves verified
re_verification: false
---

# Phase 1: Foundation Verification Report

**Phase Goal:** Establish the narwhals integration foundation — API package scaffold, dtype engine, and test infrastructure — so that all subsequent backend, schema, and validation phases have a working base to build on.
**Verified:** 2026-03-09T23:30:00Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| #  | Truth | Status | Evidence |
|----|-------|--------|----------|
| 1  | `` `pandera[narwhals]` `` extra declared in pyproject.toml with `narwhals>=2.15.0` | VERIFIED | Line 94 of pyproject.toml: `narwhals = ["narwhals >= 2.15.0"]` |
| 2  | `` `NarwhalsData(frame=lf, key='*')` `` constructs without error and its `frame` field is a `nw.LazyFrame` | VERIFIED | Smoke test confirms `isinstance(data.frame, nw.LazyFrame)` = True |
| 3  | `` `NarwhalsCheckResult` `` named tuple exists in `types.py` with four `nw.LazyFrame` fields | VERIFIED | File confirmed: `check_output`, `check_passed`, `checked_object`, `failure_cases` all typed `nw.LazyFrame` |
| 4  | `` `_to_native(nw_frame)` `` returns a native backend frame; `` `_to_native(native_frame)` `` passes through without error | VERIFIED | Smoke test: both return `pl.LazyFrame`; `pass_through=True` confirmed in source |
| 5  | Test scaffold exists and INFRA tests pass GREEN; ENGINE stubs now all pass GREEN (Plan 02 complete) | VERIFIED | `python -m pytest tests/backends/narwhals/test_narwhals_dtypes.py`: 26 passed, 0 failed, 0 xfail |
| 6  | `` `Engine.dtype(nw.Int8)` `` through `` `Engine.dtype(nw.UInt64)` `` resolve to narwhals `DataType` subclasses | VERIFIED | 8 `test_dtype_registration` parametrized cases all PASS |
| 7  | `` `Engine.dtype(nw.Float32)` ``, `` `Engine.dtype(nw.Float64)` `` resolve correctly | VERIFIED | Parametrized test passes for Float32, Float64 |
| 8  | `` `Engine.dtype(nw.Datetime('us', 'UTC'))` `` resolves via `from_parametrized_dtype` dispatch | VERIFIED | `test_datetime_parameterized` PASSES; `from_parametrized_dtype` reads `nw_dtype.time_unit` and `nw_dtype.time_zone` |
| 9  | `narwhals_engine.py` is NOT imported by any `__init__.py` or top-level pandera import | VERIFIED | Grep of all pandera `__init__.py` files: no matches. Opt-in isolation runtime check: 0 narwhals engines registered on plain `import pandera` |

**Score:** 9/9 truths verified

---

### Required Artifacts

#### Plan 01-01 Artifacts

| Artifact | Provides | Status | Details |
|----------|----------|--------|---------|
| `pyproject.toml` | narwhals optional-dependency extra | VERIFIED | Contains `narwhals = ["narwhals >= 2.15.0"]` at line 94 |
| `pandera/api/narwhals/__init__.py` | Package init | VERIFIED | Exists; contains one-line docstring `"""Narwhals API."""` |
| `pandera/api/narwhals/types.py` | `NarwhalsData`, `NarwhalsCheckResult` | VERIFIED | Both NamedTuples defined; `frame` field (not `lazyframe`); all `nw.LazyFrame` typed |
| `pandera/api/narwhals/utils.py` | `_to_native` helper | VERIFIED | Exports `_to_native`; uses `nw.to_native(frame, pass_through=True)` |
| `tests/backends/__init__.py` | Backend test package init | VERIFIED | Exists; empty package init |
| `tests/backends/narwhals/__init__.py` | Narwhals test sub-package init | VERIFIED | Exists; empty package init |
| `tests/backends/narwhals/test_narwhals_dtypes.py` | Test suite covering INFRA-02, INFRA-03, ENGINE-01/02/03 | VERIFIED | 26 tests, all pass |

#### Plan 01-02 Artifacts

| Artifact | Provides | Status | Details |
|----------|----------|--------|---------|
| `pandera/engines/narwhals_engine.py` | `DataType` base, `Engine` class, 18 dtype registrations, `coerce`/`try_coerce` | VERIFIED | All 18 dtypes registered and resolving; coerce/try_coerce functional; 355 lines, substantive implementation |

---

### Key Link Verification

#### Plan 01-01 Key Links

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `pandera/api/narwhals/types.py` | `narwhals.stable.v1` | `import narwhals.stable.v1 as nw` | VERIFIED | Line 4: `import narwhals.stable.v1 as nw` — no bare `narwhals` |
| `pandera/api/narwhals/utils.py` | `nw.to_native` | `pass_through=True` | VERIFIED | Line 13: `return nw.to_native(frame, pass_through=True)` |

#### Plan 01-02 Key Links

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `pandera/engines/narwhals_engine.py` | `pandera/api/narwhals/types.py` | lazy import inside `coerce`/`try_coerce` | VERIFIED | `from pandera.api.narwhals.types import NarwhalsData` inside method body |
| `pandera/engines/narwhals_engine.py` | `pandera/api/narwhals/utils.py` | lazy import inside `try_coerce` | VERIFIED | `from pandera.api.narwhals.utils import _to_native` inside method body |
| `pandera/engines/narwhals_engine.py` | `pandera.engines.engine.Engine` | `metaclass=engine.Engine` | VERIFIED | Line 88: `class Engine(metaclass=engine.Engine, base_pandera_dtypes=DataType)` |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| INFRA-01 | 01-01 | `narwhals>=2.15.0` in `pyproject.toml`; all imports use `narwhals.stable.v1` | SATISFIED | `pyproject.toml` line 94; all source files use `import narwhals.stable.v1 as nw` |
| INFRA-02 | 01-01 | `pandera/api/narwhals/types.py` with `NarwhalsData` named tuple | SATISFIED | File exists; `NarwhalsData` is a `NamedTuple` with `frame: nw.LazyFrame` and `key: str = "*"` |
| INFRA-03 | 01-01 | `pandera/api/narwhals/utils.py` with `_to_native()` helper | SATISFIED | File exists; `_to_native` uses `pass_through=True`; smoke test confirms both narwhals and native frames handled |
| ENGINE-01 | 01-02 | `pandera/engines/narwhals_engine.py` with `Engine` metaclass | SATISFIED | File exists; `Engine` class uses `metaclass=engine.Engine` following polars_engine.py pattern |
| ENGINE-02 | 01-02 | All narwhals dtype objects registered via `@Engine.register_dtype` | SATISFIED | 18 registrations: Int8/16/32/64, UInt8/16/32/64, Float32/64, String, Bool, Date, DateTime, Duration, Categorical, List, Struct — all 18 `test_dtype_registration` parametrized cases PASS |
| ENGINE-03 | 01-02 | `coerce()` and `try_coerce()` implemented | SATISFIED | `coerce()` uses `lf.with_columns(nw.col().cast())`, returns `nw.LazyFrame` lazy; `try_coerce()` calls `.collect()` to force evaluation, raises `ParserError` with native `failure_cases` (verified `isinstance(fc, pl.DataFrame)`) |

**Orphaned requirements:** None — all 6 Phase 1 requirements are claimed and satisfied.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | — | — | None found |

No TODO, FIXME, PLACEHOLDER, empty returns, or stub implementations detected in any phase 1 file.

---

### Human Verification Required

None — all phase 1 behaviors are programmatically verifiable. The test suite runs deterministically and all 26 tests pass.

---

### Gaps Summary

No gaps. All 9 observable truths verified, all 8 artifacts confirmed at all three levels (exists, substantive, wired), all 5 key links confirmed, all 6 requirements satisfied, no orphaned requirements, no blocker anti-patterns.

---

## Detailed Test Results

```
tests/backends/narwhals/test_narwhals_dtypes.py::test_narwhals_data_type            PASSED
tests/backends/narwhals/test_narwhals_dtypes.py::test_narwhals_check_result_fields  PASSED
tests/backends/narwhals/test_narwhals_dtypes.py::test_to_native_narwhals_frame      PASSED
tests/backends/narwhals/test_narwhals_dtypes.py::test_to_native_already_native      PASSED
tests/backends/narwhals/test_narwhals_dtypes.py::test_engine_dtype                  PASSED
tests/backends/narwhals/test_narwhals_dtypes.py::test_dtype_registration[Int8]      PASSED
tests/backends/narwhals/test_narwhals_dtypes.py::test_dtype_registration[Int16]     PASSED
tests/backends/narwhals/test_narwhals_dtypes.py::test_dtype_registration[Int32]     PASSED
tests/backends/narwhals/test_narwhals_dtypes.py::test_dtype_registration[Int64]     PASSED
tests/backends/narwhals/test_narwhals_dtypes.py::test_dtype_registration[UInt8]     PASSED
tests/backends/narwhals/test_narwhals_dtypes.py::test_dtype_registration[UInt16]    PASSED
tests/backends/narwhals/test_narwhals_dtypes.py::test_dtype_registration[UInt32]    PASSED
tests/backends/narwhals/test_narwhals_dtypes.py::test_dtype_registration[UInt64]    PASSED
tests/backends/narwhals/test_narwhals_dtypes.py::test_dtype_registration[Float32]   PASSED
tests/backends/narwhals/test_narwhals_dtypes.py::test_dtype_registration[Float64]   PASSED
tests/backends/narwhals/test_narwhals_dtypes.py::test_dtype_registration[String]    PASSED
tests/backends/narwhals/test_narwhals_dtypes.py::test_dtype_registration[Boolean]   PASSED
tests/backends/narwhals/test_narwhals_dtypes.py::test_dtype_registration[Date]      PASSED
tests/backends/narwhals/test_narwhals_dtypes.py::test_dtype_registration[Datetime]  PASSED
tests/backends/narwhals/test_narwhals_dtypes.py::test_dtype_registration[Duration]  PASSED
tests/backends/narwhals/test_narwhals_dtypes.py::test_dtype_registration[Categorical] PASSED
tests/backends/narwhals/test_narwhals_dtypes.py::test_dtype_registration[List]      PASSED
tests/backends/narwhals/test_narwhals_dtypes.py::test_dtype_registration[Struct]    PASSED
tests/backends/narwhals/test_narwhals_dtypes.py::test_coerce_returns_lazyframe      PASSED
tests/backends/narwhals/test_narwhals_dtypes.py::test_try_coerce_raises_on_invalid_cast PASSED
tests/backends/narwhals/test_narwhals_dtypes.py::test_datetime_parameterized        PASSED

26 passed in 0.97s
```

## Commit Verification

Commits documented in summaries are confirmed in git log:

| Commit | Plan | Task |
|--------|------|------|
| `7df9650` | 01-01 | Add narwhals extra + create API package |
| `5841155` | 01-01 | Create test scaffold |
| `fc80e9e` | 01-02 | Implement narwhals dtype engine |

---

_Verified: 2026-03-09T23:30:00Z_
_Verifier: Claude (gsd-verifier)_
