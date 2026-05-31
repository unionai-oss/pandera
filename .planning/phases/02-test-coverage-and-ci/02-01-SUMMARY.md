---
plan: 02-01
phase: 02-test-coverage-and-ci
status: complete
completed: 2026-05-11
requirements: [TEST-02]
---

# Plan 02-01: Apply xfail Markers to PySpark Tests

## What Was Built

Applied `@pytest.mark.xfail(condition=CONFIG.use_narwhals_backend, strict=True, reason=...)` markers to 11 test sites across 4 PySpark test files, documenting expected SQL-lazy limitations under the Narwhals backend.

## Key Files

### Created
_(none)_

### Modified
- `tests/pyspark/test_pyspark_config.py` — Added CONFIG import + 5 xfail markers on all TestPanderaConfig methods (config dict assertions hardcode use_narwhals_backend=False)
- `tests/pyspark/test_pyspark_check.py` — Added CONFIG import + 4 xfail markers (TestCustomCheck.test_extension + test_extension_dataframe_model: PysparkDataframeColumnObject API incompatible with narwhals; TestUniqueValuesEqCheck.test_unique_values_eq_check + test_failed_unaccepted_datatypes: unique_values_eq not registered for narwhals)
- `tests/pyspark/test_pyspark_container.py` — Added CONFIG to import + 1 xfail marker on test_pyspark_sample (sample= unsupported in narwhals)
- `tests/pyspark/test_pyspark_decorators.py` — Added CONFIG to import + 1 xfail marker on test_cache_dataframe_settings (narwhals bypasses PySpark caching decorators)

## Commits
- `e86f18d7` feat(02-01): apply xfail markers to test_pyspark_config.py
- `98f3c609` feat(02-01): apply xfail markers to test_pyspark_check, container, decorators

## Decisions Made
- All markers use `condition=CONFIG.use_narwhals_backend` (D-03 — not os.getenv)
- All markers use `strict=True` (D-04)
- test_cache_dataframe_requirements was NOT xfail-marked (expected to pass per RESEARCH.md A1)
- xfail markers placed as outermost decorator where stacked with @pytest.mark.parametrize

## Deviations
- The executor agent committed test_pyspark_config.py to the main branch (not its worktree) due to a stream idle timeout; the orchestrator recovered by committing the remaining 3 files directly.

## Self-Check: PASSED

- 11 xfail markers total (5+4+1+1) ✓
- All use condition=CONFIG.use_narwhals_backend ✓
- All use strict=True ✓
- All 4 files import CONFIG ✓
- No os.getenv() xfail conditions ✓
- All 4 files parse as valid Python ✓
