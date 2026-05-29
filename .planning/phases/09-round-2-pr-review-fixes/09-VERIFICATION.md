---
phase: 09-round-2-pr-review-fixes
verified: 2026-05-29T00:00:00Z
status: passed
score: 8/8 must-haves verified
overrides_applied: 0
---

# Phase 09: Round 2 PR Review Fixes Verification Report

**Phase Goal:** Resolve all Round 2 PR review findings (B-01, M-01 through M-07) — production code correctness fixes, exception guard narrowing, type deduplication, and test hygiene improvements.
**Verified:** 2026-05-29
**Status:** PASSED
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|---------|
| 1 | ColumnBackend.validate() does not unconditionally convert nw.DataFrame to nw.LazyFrame for SQL-lazy frames (PySpark) | VERIFIED | `components.py:68`: `if isinstance(check_lf, nw.DataFrame) and not _is_sql_lazy(check_lf):` — `_is_sql_lazy` already imported on line 14 |
| 2 | No `except Exception:` guards remain in the three production PySpark import sites | VERIFIED | `grep -n "except Exception" types.py register.py pyspark_sql_accessor.py` returns nothing (exit=1) |
| 3 | PySparkDtypeInputTypes Union has no duplicate `type` entry | VERIFIED | `grep -cE "^\s*type,\s*$" pandera/api/pyspark/types.py` returns 1 |
| 4 | supported_types() has no dead try/except ImportError block around the PySparkConnectDataFrame append | VERIFIED | `types.py:101-104`: direct list literal `[PySparkSQLDataFrame, PySparkConnectDataFrame]`, no try/except; `grep -c "# pragma: no cover" types.py` returns 0 |
| 5 | No `if CONFIG.use_narwhals_backend:` inline branches remain in test_pyspark_dtypes.py | VERIFIED | `grep -vE '^\s*#' test_pyspark_dtypes.py \| grep -cE "if (not )?CONFIG.use_narwhals_backend"` returns 0; module-level `pytestmark` list with `pytest.mark.skipif(CONFIG.use_narwhals_backend, ...)` added |
| 6 | tests/narwhals/conftest.py owns both `_spark_env_vars` (autouse) and `spark` session fixtures; the two test modules no longer define them | VERIFIED | `grep -cE "^def _spark_env_vars"` returns 0/0/1 for test_e2e/test_arch03/conftest; `grep -cE "^def spark\b"` returns 0/0/1; Phase 7 yield regression guard present (yield on both branches of `_spark_env_vars`) |
| 7 | TestPanderaConfig._cmp_errors static wrapper is gone; all callers use the module-level _cmp_errors from tests/pyspark/conftest.py | VERIFIED | `grep -c "def _cmp_errors" test_pyspark_config.py` returns 0; `grep -c "self._cmp_errors" test_pyspark_config.py` returns 0; `grep -cE "(^|\s)_cmp_errors\(" test_pyspark_config.py` returns 8 |
| 8 | test_pyspark_decorators.py no longer calls pytest.xfail() inline; the cache_enabled+use_narwhals_backend skip is expressed as a parametrize-level mark | VERIFIED | `grep -c "pytest.xfail(" test_pyspark_decorators.py` returns 0; `grep -c "pytest.mark.xfail(" test_pyspark_decorators.py` returns 2; `grep -c "pytest.param(" test_pyspark_decorators.py` returns 2 |

**Score:** 8/8 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `pandera/backends/narwhals/components.py` | ColumnBackend.validate guarded with `_is_sql_lazy()` | VERIFIED | Line 68 contains `and not _is_sql_lazy(check_lf)`; import on line 14 unchanged |
| `pandera/api/pyspark/types.py` | ImportError-only guard; deduplicated PySparkDtypeInputTypes; no dead handler in supported_types() | VERIFIED | `except ImportError:` at line 23; `type,` appears once at line 69; `supported_types()` uses direct list literal |
| `pandera/backends/pyspark/register.py` | ImportError-only guard for pyspark_connect import | VERIFIED | `except ImportError:` at line 16 |
| `pandera/accessors/pyspark_sql_accessor.py` | ImportError-only guard for register_connect_dataframe_accessor | VERIFIED | `except ImportError:` at line 159 |
| `tests/narwhals/conftest.py` | Shared `_spark_env_vars` autouse fixture and `spark` session fixture | VERIFIED | Both fixtures present with correct decorators; Phase 7 yield regression guard intact |
| `tests/pyspark/test_pyspark_dtypes.py` | Dtype tests free of inline use_narwhals_backend branching | VERIFIED | `pytestmark` is a list containing both `pytest.mark.parametrize` and `pytest.mark.skipif`; `df.pandera.schema == pandera_schema` assertion unconditional at line 42 |
| `tests/pyspark/test_pyspark_config.py` | Test class that calls module-level `_cmp_errors` directly | VERIFIED | 8 bare `_cmp_errors(...)` calls; import `from tests.pyspark.conftest import _cmp_errors, spark_df` present |
| `tests/pyspark/test_pyspark_decorators.py` | Parametrize-level xfail mark for the cache_enabled+narwhals combination | VERIFIED | Two `pytest.param(...)` entries with `marks=pytest.mark.xfail(CONFIG.use_narwhals_backend, ...)` and `strict=False` |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `components.py:validate` | `pandera.api.narwhals.utils:_is_sql_lazy` | import on line 14 | WIRED | `_is_sql_lazy` imported in the existing import block; `_is_sql_lazy(check_lf)` called on line 68 |
| `tests/narwhals/test_e2e.py` | `tests/narwhals/conftest.py` | pytest auto-discovery | WIRED | `def spark` and `def _spark_env_vars` not defined in test_e2e.py; conftest defines both |
| `tests/pyspark/test_pyspark_config.py` | `tests/pyspark/conftest.py:_cmp_errors` | module-level import | WIRED | `from tests.pyspark.conftest import _cmp_errors, spark_df` at line 21; 8 direct call sites |

### Data-Flow Trace (Level 4)

Not applicable — this phase modifies guard logic, exception handling, type annotations, and test fixtures. No dynamic data rendering paths introduced.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| components.py imports cleanly under narwhals backend | `PANDERA_USE_NARWHALS_BACKEND=True python -c "import pandera.backends.narwhals.components"` | OK | PASS |
| Three PySpark production files import cleanly | `python -c "import pandera.api.pyspark.types; import pandera.backends.pyspark.register; import pandera.accessors.pyspark_sql_accessor"` | OK | PASS |

### Probe Execution

Step 7c: SKIPPED — no `scripts/*/tests/probe-*.sh` files declared in PLAN or present in this phase.

### Requirements Coverage

No requirement IDs declared for this phase (requirements: [] in both PLANs). Phase goal is tracked by findings B-01, M-01 through M-07. All 8 findings verified above.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `pandera/backends/narwhals/components.py` | 326 | `TODO` comment | Info | Pre-existing TODO about schema construction in a future milestone; not introduced by this phase (commit c833f72f changed only line 68). No new debt added. |

No `TBD`, `FIXME`, or `XXX` markers in any file modified by this phase. The `TODO` on line 326 of `components.py` predates phase 09 (git log confirms c833f72f only changed line 68); it does not constitute new unresolved debt.

### Human Verification Required

None — all must-haves are verifiable via static analysis and import checks. No visual output, real-time behavior, or external service integration introduced.

### Gaps Summary

No gaps. All 8 observable truths are verified with direct code evidence. All 7 commits exist in git history. All artifacts are substantive (not stubs) and correctly wired. No new anti-patterns introduced by this phase.

---

_Verified: 2026-05-29_
_Verifier: Claude (gsd-verifier)_
