---
phase: 06-test-coverage-and-minor-fixes
verified: 2026-05-25T00:00:00Z
status: passed
score: 6/6 must-haves verified
overrides_applied: 0
---

# Phase 6: Test Coverage and Minor Fixes — Verification Report

**Phase Goal:** PySpark narwhals coverage exists in `tests/narwhals/test_e2e.py` and all minor issues from the PR review are resolved
**Verified:** 2026-05-25
**Status:** PASSED
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `tests/narwhals/test_e2e.py` includes a PySpark section with backend registration assertion, return-type preservation, passing/failing built-in check with errors inspection, nullable, and unique behavior | VERIFIED | Six `@pyspark_only` test functions present at lines 755-826: `test_narwhals_backend_registered_for_pyspark_dataframe`, `test_pyspark_dataframe_returns_pyspark_dataframe`, `test_pyspark_builtin_check_passes`, `test_pyspark_builtin_check_fails`, `test_pyspark_nullable_false_fails`, `test_pyspark_unique_constraint_fails`. All use accessor-based error contract (`df_out.pandera.errors`). 4 uses of `df_out.pandera.errors` confirmed. |
| 2 | The CI Python version exclusion for PySpark (3.12, 3.13) has a comment explaining PySpark's maximum supported Python version | VERIFIED | Line 347 of `.github/workflows/ci-tests.yml`: `# PySpark's maximum supported Python version is 3.11; 3.12 and 3.13 are not yet on the official PySpark support matrix.` present immediately above the `exclude:` block. |
| 3 | `container.py` emits backend-neutral "column 'X' not found" message; corresponding xfail in `test_ibis_container.py` is removed | VERIFIED | `pandera/backends/narwhals/container.py` line 594 emits `f"column '{colname}' not found"`. `pandera/backends/ibis/container.py` line 386 emits `f"column '{colname}' not found. "`. `test_column_absent_error` in `tests/ibis/test_ibis_container.py` has no `@pytest.mark.xfail` decorator — match regex is `"column 'int_col' not found"`. |
| 4 | `test_pyspark_narwhals_register.py::test_pyspark_narwhals_activated_when_opted_in` asserts all three backends are registered | VERIFIED | Lines 68–74: `assert isinstance(backend, NarwhalsDataFrameSchemaBackend)`, `assert isinstance(column_backend, NarwhalsColumnBackend)`, `assert check_registry_key in Check.BACKEND_REGISTRY`, `assert Check.BACKEND_REGISTRY[check_registry_key] is NarwhalsCheckBackend`. |
| 5 | Stacked `@pytest.mark.xfail` decorators in `test_pyspark_model.py::test_registered_dataframemodel_checks` are combined into a single conditional xfail | VERIFIED | `grep -B10 "def test_registered_dataframemodel_checks"` returns exactly 1 `@pytest.mark.xfail`. Single decorator at lines 551-558: `raises=Exception, strict=False`. |
| 6 | The `supported_types()` double-append of `PySparkSQLDataFrame` is fixed | VERIFIED | `grep -c "table_types.append(PySparkSQLDataFrame)"` returns 0. `pandera/api/pyspark/types.py` lines 102–112: `table_types = [PySparkSQLDataFrame]` then `try: table_types.append(PySparkConnectDataFrame)` — only one append, initialised via list literal. |

**Score:** 6/6 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `tests/narwhals/test_e2e.py` | PySpark section with 6 tests | VERIFIED | 6 tests decorated `@pyspark_only`; `pandera.pyspark` imported at line 63 |
| `tests/narwhals/conftest.py` | `register_pyspark_backends` called inside autouse fixture | VERIFIED | Lines 44-52: `try: import pyspark.sql; from pandera.backends.pyspark.register import register_pyspark_backends; register_pyspark_backends.cache_clear(); register_pyspark_backends(); except ImportError: pass` |
| `.github/workflows/ci-tests.yml` | Comment on pyspark exclude block | VERIFIED | Line 347: comment naming 3.11 as max supported Python version |
| `pandera/backends/narwhals/container.py` | "not found" message | VERIFIED | Line 594: `f"column '{colname}' not found"` |
| `pandera/backends/ibis/container.py` | "not found" message | VERIFIED | Line 386: `f"column '{colname}' not found. "` |
| `tests/ibis/test_ibis_container.py` | xfail removed, match regex updated | VERIFIED | No xfail on `test_column_absent_error`; match is `"column 'int_col' not found"` |
| `tests/pyspark/test_pyspark_narwhals_register.py` | All three backends asserted | VERIFIED | `isinstance(backend, NarwhalsDataFrameSchemaBackend)`, `isinstance(column_backend, NarwhalsColumnBackend)`, `Check.BACKEND_REGISTRY[key] is NarwhalsCheckBackend` |
| `tests/pyspark/test_pyspark_model.py` | Single xfail decorator | VERIFIED | 1 `@pytest.mark.xfail` immediately before `test_registered_dataframemodel_checks` |
| `pandera/api/pyspark/types.py` | No duplicate PySparkSQLDataFrame append | VERIFIED | `table_types.append(PySparkSQLDataFrame)` count = 0 |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `tests/narwhals/test_e2e.py PySpark section` | `pandera.backends.pyspark.register.register_pyspark_backends` | Module-scope conftest fixture re-registers pyspark backends | WIRED | `conftest.py` autouse fixture calls `register_pyspark_backends.cache_clear()` then `register_pyspark_backends()` inside `try/except ImportError` |
| `tests/narwhals/test_e2e.py PySpark section` | `NarwhalsCheckBackend` | `Check.get_backend(pyspark.sql.DataFrame)` | WIRED | `test_narwhals_backend_registered_for_pyspark_dataframe` calls `Check.get_backend(spark.createDataFrame(...))` and asserts `backend is NarwhalsCheckBackend` |
| `tests/narwhals/test_e2e.py PySpark section` | `df_out.pandera.errors` | Accessor-based error contract (4 assertions) | WIRED | `test_pyspark_builtin_check_passes`, `test_pyspark_builtin_check_fails`, `test_pyspark_nullable_false_fails`, `test_pyspark_unique_constraint_fails` all assert on `.pandera.errors` |
| `pandera/backends/narwhals/container.py` | `tests/ibis/test_ibis_container.py` | `"column '.*' not found"` error message | WIRED | Both backends emit "not found"; test asserts `match="column 'int_col' not found"` without xfail |
| `pandera/backends/ibis/container.py` | `tests/ibis/test_ibis_container.py` | `"column '.*' not found"` error message | WIRED | Ibis container line 386 emits `"not found."` which matches the regex `"not found"` |

### Behavioral Spot-Checks

Step 7b skipped for this verification — tests run under PySpark require a live SparkSession and JVM; running the test suite is out of scope for static verification. Commit hashes all verified present in git history: `070c838b`, `cbdd9dab`, `d010babd`, `7b14c75b`, `029434f2`, `bb8b3829`, `6d65a067`.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|---------|
| TEST-E2E-01 | Plan 01 | PySpark e2e section in `tests/narwhals/test_e2e.py` | SATISFIED | 6 pyspark tests present with full coverage of all subcriteria |
| NITS-01 | Plan 02 | Minor pre-merge nits (CI comment, error messages, registration test, stacked xfail, duplicate append) | SATISFIED | All 5 sub-items verified individually (truths 2–6) |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `pandera/api/pyspark/types.py` | 104-108 | `try: table_types.append(PySparkConnectDataFrame) except ImportError: pass` — `ImportError` can never be raised because `PySparkConnectDataFrame` is imported at module level unconditionally | INFO | Pre-existing dead-code `except` clause; does not affect correctness; duplicate append (the real bug) is fixed. Out of scope for this phase. |

No `TBD`, `FIXME`, or `XXX` markers found in phase-modified files.

### Human Verification Required

None — all observable truths are verifiable from static code inspection and git history.

### Gaps Summary

No gaps. All 6 success criteria verified against actual code. All 7 SUMMARY-claimed commits exist in git history and their diffs match the described changes.

---

_Verified: 2026-05-25_
_Verifier: Claude (gsd-verifier)_
