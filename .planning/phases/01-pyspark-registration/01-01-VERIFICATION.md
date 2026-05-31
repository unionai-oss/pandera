---
phase: 01-pyspark-registration
verified: 2026-05-10T00:00:00Z
status: passed
score: 6/6
overrides_applied: 0
re_verification: null
---

# Phase 1: PySpark Registration — Verification Report

**Phase Goal:** Users can activate the Narwhals backend for PySpark by setting `PANDERA_USE_NARWHALS_BACKEND=True`, with existing native PySpark behavior unchanged when the flag is off
**Verified:** 2026-05-10
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| #  | Truth                                                                                                                                                | Status     | Evidence                                                                                                                                                                   |
|----|------------------------------------------------------------------------------------------------------------------------------------------------------|------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1  | Setting `CONFIG.use_narwhals_backend=True` causes `register_pyspark_backends()` to register `NarwhalsDataFrameSchemaBackend` for `pyspark_sql.DataFrame` | VERIFIED   | `register.py` line 44: `if CONFIG.use_narwhals_backend:` branch; `DataFrameSchema.register_backend(pyspark_sql.DataFrame, DataFrameSchemaBackend)` at line 59; test `test_pyspark_narwhals_activated_when_opted_in` passes |
| 2  | Setting `CONFIG.use_narwhals_backend=False` leaves native `pandera.backends.pyspark.container.DataFrameSchemaBackend` registered                      | VERIFIED   | `else` branch at line 68 preserves native `PySparkDataFrameSchemaBackend` registration (line 90-92); test `test_pyspark_native_unchanged_when_flag_off` passes              |
| 3  | When `PYSPARK_CONNECT_AVAILABLE` is True and the narwhals branch runs, `pyspark_connect.DataFrame` is also registered with Narwhals backends          | VERIFIED   | `register.py` lines 64-67: `if PYSPARK_CONNECT_AVAILABLE:` block inside narwhals branch registers `DataFrameSchemaBackend`, `ColumnBackend`, `NarwhalsCheckBackend` for `pyspark_connect.DataFrame`; test `test_pyspark_connect_narwhals_activated_when_opted_in` correctly skips when grpcio-status absent |
| 4  | Calling `register_pyspark_backends()` twice does not raise (`lru_cache` guarantees idempotency)                                                        | VERIFIED   | `@lru_cache` decorator at line 21; test `test_pyspark_register_is_idempotent` passes                                                                                      |
| 5  | When the narwhals branch runs, `pandera.backends.narwhals.builtin_checks` is imported for its side effect                                              | VERIFIED   | `register.py` line 54: `import pandera.backends.narwhals.builtin_checks  # noqa: F401` inside the `if CONFIG.use_narwhals_backend:` block                                |
| 6  | When the narwhals branch runs, `_patch_numpy2()` is NOT invoked                                                                                       | VERIFIED   | `_patch_numpy2()` call at line 71 is exclusively inside the `else:` branch; grep confirms only 1 occurrence, within the native path                                       |

**Score:** 6/6 truths verified

### Required Artifacts

| Artifact                                          | Expected                                                                            | Status     | Details                                                                                                                         |
|---------------------------------------------------|-------------------------------------------------------------------------------------|------------|---------------------------------------------------------------------------------------------------------------------------------|
| `pandera/backends/pyspark/register.py`            | Conditional narwhals/native registration; contains `if CONFIG.use_narwhals_backend:` | VERIFIED   | File exists, 107 lines, fully substantive implementation with both branches; syntactically valid (`ast.parse` exits 0)          |
| `tests/narwhals/test_container.py`                | 4 new pyspark test functions including `test_pyspark_narwhals_activated_when_opted_in` | VERIFIED   | All 4 test functions present at lines 250, 281, 310, 348; 3 pass, 1 correctly skips in test environment                       |

### Key Link Verification

| From                                              | To                                         | Via                                              | Status     | Details                                                                                       |
|---------------------------------------------------|--------------------------------------------|--------------------------------------------------|------------|-----------------------------------------------------------------------------------------------|
| `pandera/backends/pyspark/register.py`            | `pandera.config.CONFIG.use_narwhals_backend` | `from pandera.config import CONFIG; if CONFIG.use_narwhals_backend:` | VERIFIED | Line 43: `from pandera.config import CONFIG`; line 44: `if CONFIG.use_narwhals_backend:`     |
| `register.py` (narwhals branch)                   | `pandera.backends.narwhals.{checks,components,container,builtin_checks}` | imports + `register_backend()` calls | VERIFIED | Lines 54-57 import all four; `DataFrameSchema.register_backend`, `Column.register_backend`, `Check.register_backend` called lines 59-62 |
| `register.py` (else branch)                       | `pandera._patch_numpy2._patch_numpy2` and native pyspark backends | preserved existing native imports + registrations | VERIFIED | Lines 69-106: `_patch_numpy2()` called, all native `PySparkCheckBackend`, `ColumnSchemaBackend`, `PySparkColumnBackend`, `PySparkDataFrameSchemaBackend` imported and registered |

### Data-Flow Trace (Level 4)

Not applicable — this phase delivers registration wiring (side-effect code), not a component that renders dynamic data. The "data" is the `BACKEND_REGISTRY` dict populated at call time; correctness is proven by the test assertions.

### Behavioral Spot-Checks

| Behavior                                       | Command                                                                                    | Result                                     | Status  |
|------------------------------------------------|--------------------------------------------------------------------------------------------|--------------------------------------------|---------|
| All 4 pyspark tests pass (3 pass, 1 skip)      | `pytest tests/narwhals/test_container.py -k "pyspark" -v`                                 | 3 passed, 1 skipped                        | PASS    |
| `register.py` is syntactically valid Python    | `python3 -c "import ast; ast.parse(open('pandera/backends/pyspark/register.py').read())"` | No error                                   | PASS    |
| `test_container.py` is syntactically valid     | `python3 -c "import ast; ast.parse(open('tests/narwhals/test_container.py').read())"`     | No error                                   | PASS    |

### Requirements Coverage

| Requirement | Source Plan | Description                                                                                                                                             | Status    | Evidence                                                                                                                               |
|-------------|-------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|-----------|----------------------------------------------------------------------------------------------------------------------------------------|
| REG-01      | 01-01-PLAN  | `register_pyspark_backends()` conditionally registers `NarwhalsCheckBackend`, `ColumnBackend`, and `DataFrameSchemaBackend` for `pyspark_sql.DataFrame` (and `pyspark_connect.DataFrame` if available) when `PANDERA_USE_NARWHALS_BACKEND=True`; the `else` branch keeps existing native registrations untouched | SATISFIED | Implementation present in `register.py` lines 44-106; all three narwhals classes registered for `pyspark_sql.DataFrame` (lines 59-62), `pyspark_connect.DataFrame` guarded by `PYSPARK_CONNECT_AVAILABLE` (lines 64-67); native registrations unchanged in `else` branch (lines 68-106) |

No REQUIREMENTS.md requirement IDs assigned to Phase 1 other than REG-01. All other requirements (TEST-01, TEST-02, TEST-03, CI-01, DOCS-01) are mapped to Phase 2 or Phase 3 per the traceability table — they are explicitly deferred and not orphaned.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None | — | — | — | — |

No TODOs, FIXMEs, placeholders, empty implementations, or stubs found in either file. The `try/except Exception: PYSPARK_CONNECT_AVAILABLE = False` pattern at lines 13-18 in `register.py` and the fallback in `types.py` lines 18-28 are intentional defensive guards, not stubs.

### Human Verification Required

None. All success criteria for this phase are verifiable programmatically. The behavioral spot-checks (pytest run) confirm the registration mechanics work correctly.

### Additional Notes

**Deviation: BACKEND_REGISTRY save/restore in tests.** The PLAN specified `request.addfinalizer(register_pyspark_backends.cache_clear)` plus `cache_clear()` before each test. During execution, a further deviation was necessary: because `register_backend()` uses first-registration-wins semantics, clearing the lru_cache alone is insufficient to achieve test isolation — the BACKEND_REGISTRY dict persists across tests in the same process. The implementation correctly saves and restores the `(PySparkDataFrameSchema, pyspark_sql.DataFrame)` entry via `BACKEND_REGISTRY.pop()` + `addfinalizer` restore. This is a valid, substantive implementation — not a stub — and correctly proves the goal.

**Deviation: `get_backend(check_type=...)` kwarg.** The PLAN prescribed `get_backend(pyspark_sql.DataFrame)` as a positional arg. Creating a PySpark DataFrame requires a SparkSession, so the test correctly uses `check_type=pyspark_sql.DataFrame` instead. This does not affect the goal.

**Bug fixes applied during execution.** Two import-crash bugs were fixed:
1. `pandera/backends/pyspark/register.py` module-level `pyspark.sql.connect` import wrapped in `try/except Exception` (lines 13-18)
2. `pandera/api/pyspark/types.py` `pyspark.sql.connect.dataframe` import wrapped similarly (lines 17-28)
Both are out-of-scope bugs that had to be fixed to make any test run at all. They are real fixes, not stubs.

### Gaps Summary

No gaps. All 6 observable truths are VERIFIED. Both artifacts exist and are substantive implementations with complete wiring. The single requirement REG-01 is fully satisfied. All pyspark tests pass (connect variant correctly skips due to absent grpcio-status in test environment, which is acceptable per the PLAN acceptance criteria).

---

_Verified: 2026-05-10_
_Verifier: Claude (gsd-verifier)_
