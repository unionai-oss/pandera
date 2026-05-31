---
phase: 09-round-2-pr-review-fixes
plan: "02"
subsystem: test-quality
tags: [pyspark, narwhals, pytest, fixtures, test-hygiene]
dependency_graph:
  requires: [09-01]
  provides: [M-02-resolved, M-03-resolved, M-04-resolved, M-07-resolved]
  affects: [tests/pyspark/test_pyspark_dtypes.py, tests/narwhals/conftest.py, tests/narwhals/test_e2e.py, tests/narwhals/test_arch03_schema_driven_dispatch.py, tests/pyspark/test_pyspark_config.py, tests/pyspark/test_pyspark_decorators.py]
tech_stack:
  added: []
  patterns: [pytest.mark.skipif module-level, pytest.param with marks, fixture consolidation in conftest]
key_files:
  created: []
  modified:
    - tests/pyspark/test_pyspark_dtypes.py
    - tests/narwhals/conftest.py
    - tests/narwhals/test_e2e.py
    - tests/narwhals/test_arch03_schema_driven_dispatch.py
    - tests/pyspark/test_pyspark_config.py
    - tests/pyspark/test_pyspark_decorators.py
decisions:
  - "Use list pytestmark to combine existing spark_session parametrize with new skipif in test_pyspark_dtypes.py — avoids losing spark_session parametrization"
  - "Remove os and SparkSession imports from test_arch03 after fixture removal — they were only needed by the removed fixtures"
  - "strict=False on xfail marks in test_pyspark_decorators.py to mirror prior runtime pytest.xfail() non-strict semantics"
metrics:
  duration: "4m"
  completed: "2026-05-29T15:02:32Z"
  tasks_completed: 4
  files_changed: 6
---

# Phase 09 Plan 02: Round 2 PR Review Test Fixes Summary

Resolved four test-hygiene findings from Round 2 PR review (M-02, M-03, M-04, M-07): removed inline `CONFIG.use_narwhals_backend` branches from dtype tests, consolidated duplicated PySpark fixtures into narwhals conftest, removed redundant static wrapper for `_cmp_errors`, and converted an inline `pytest.xfail()` to a parametrize-level `pytest.mark.xfail`.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Replace inline use_narwhals_backend branch in test_pyspark_dtypes.py | f0899afb | tests/pyspark/test_pyspark_dtypes.py |
| 2 | Move _spark_env_vars and spark fixtures into tests/narwhals/conftest.py | bd4fcb21 | tests/narwhals/conftest.py, tests/narwhals/test_e2e.py, tests/narwhals/test_arch03_schema_driven_dispatch.py |
| 3 | Remove TestPanderaConfig._cmp_errors wrapper | fb844f27 | tests/pyspark/test_pyspark_config.py |
| 4 | Convert inline pytest.xfail() to parametrize-level mark | 98dbfdeb | tests/pyspark/test_pyspark_decorators.py |

## What Was Built

### M-02: Removed inline CONFIG.use_narwhals_backend branches from test_pyspark_dtypes.py

Replaced the inline `if not CONFIG.use_narwhals_backend: assert df.pandera.schema == pandera_schema` guard and the `if CONFIG.use_narwhals_backend: df = df.sparkSession.createDataFrame(...)` workaround with a module-level `pytest.mark.skipif(CONFIG.use_narwhals_backend, reason=...)` added to the `pytestmark` list. The `pytestmark` was changed from a bare `pytest.mark.parametrize(...)` to a list containing both the existing `spark_session` parametrize and the new `skipif`. The `df.pandera.schema == pandera_schema` assertion now runs unconditionally on the native PySpark backend.

### M-03: Consolidated duplicated _spark_env_vars and spark fixtures

Both `_spark_env_vars` (autouse, function-scoped) and `spark` (module-scoped) fixtures were defined identically in `tests/narwhals/test_e2e.py` and `tests/narwhals/test_arch03_schema_driven_dispatch.py`. They are now defined once in `tests/narwhals/conftest.py` with the required `os`, `HAS_PYSPARK`, and `SparkSession` imports. The Phase 7 regression guard (yield on all branches of `_spark_env_vars`) is preserved. The stale comment in test_arch03 saying "narwhals conftest.py does not define a spark fixture" was removed. The now-unused `os` and `SparkSession` imports in test_arch03 were also cleaned up.

### M-04: Removed TestPanderaConfig._cmp_errors static wrapper

`TestPanderaConfig` had a `@staticmethod _cmp_errors` that delegated to the module-level `_cmp_errors` from `tests/pyspark/conftest.py`. The static method was deleted and all 8 `self._cmp_errors(...)` call sites were replaced with bare `_cmp_errors(...)` calls. The module-level import `from tests.pyspark.conftest import _cmp_errors, spark_df` was already present.

### M-07: Converted inline pytest.xfail() to parametrize-level pytest.mark.xfail

`test_cache_dataframe_settings` had an inline `if CONFIG.use_narwhals_backend and cache_enabled: pytest.xfail(...)` block. The two `cache_enabled=True` parametrize entries are now wrapped in `pytest.param(..., marks=pytest.mark.xfail(CONFIG.use_narwhals_backend, reason=..., strict=False))`. The inline block was deleted. `strict=False` preserves the non-strict semantics of the prior runtime `pytest.xfail()` call.

## Deviations from Plan

### Auto-decided: List pytestmark in test_pyspark_dtypes.py

**Found during:** Task 1
**Issue:** The plan said to replace the existing `pytestmark` with a new `pytest.mark.skipif`. But the file already had `pytestmark = pytest.mark.parametrize("spark_session", ...)` which parametrizes all tests with `spark_session`. Replacing it would have caused fixture-not-found errors at collection time.
**Fix:** Changed `pytestmark` to a list containing both the existing parametrize mark and the new skipif mark.
**Files modified:** tests/pyspark/test_pyspark_dtypes.py
**Rule:** Rule 1 (would break tests) — applied automatically, no user permission needed.

### Auto-decided: Remove unused imports from test_arch03 after fixture removal

**Found during:** Task 2
**Issue:** After removing the `_spark_env_vars` and `spark` fixtures from test_arch03, the `import os` and `from pyspark.sql import SparkSession` imports were no longer referenced in that file.
**Fix:** Removed both unused imports.
**Files modified:** tests/narwhals/test_arch03_schema_driven_dispatch.py
**Rule:** Rule 2 (dead code after refactor) — applied automatically.

## Known Stubs

None — all changes are test refactors with no data or placeholder values.

## Threat Flags

None — test-only changes; no production code or security-relevant surface modified.

## Self-Check: PASSED

- FOUND: tests/pyspark/test_pyspark_dtypes.py
- FOUND: tests/narwhals/conftest.py
- FOUND: tests/narwhals/test_e2e.py
- FOUND: tests/narwhals/test_arch03_schema_driven_dispatch.py
- FOUND: tests/pyspark/test_pyspark_config.py
- FOUND: tests/pyspark/test_pyspark_decorators.py
- FOUND: .planning/phases/09-round-2-pr-review-fixes/09-02-SUMMARY.md
- FOUND commit f0899afb (Task 1)
- FOUND commit bd4fcb21 (Task 2)
- FOUND commit fb844f27 (Task 3)
- FOUND commit 98dbfdeb (Task 4)
