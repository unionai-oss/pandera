---
phase: 01-structural-cleanup
plan: "05"
subsystem: narwhals-checks
tags: [native-checks, polars-series, polars-dataframe, regression-tests, checks-01]
requirements: [CHECKS-01]

dependency_graph:
  requires: []
  provides: [native-polars-check-normalization, checks-01-regression-tests]
  affects:
    - pandera/backends/narwhals/checks.py
    - tests/backends/narwhals/test_e2e.py

tech_stack:
  added: []
  patterns:
    - "Wide-table approach: attach boolean mask to original frame as CHECK_OUTPUT_KEY column"
    - "nw.from_native(frame.with_columns(...)) for wrapping polars-native check output"

key_files:
  created: []
  modified:
    - pandera/backends/narwhals/checks.py
    - tests/backends/narwhals/test_e2e.py

decisions:
  - "Wide-table approach required (not narrow): postprocess_lazyframe_output expects original cols + CHECK_OUTPUT_KEY, not just a single-column frame"
  - "Collect LazyFrame before with_columns() because pl.Series is eager and can't be attached to a LazyFrame directly"
  - "pl.Series and pl.DataFrame handled together in one isinstance check — both follow same wide-table attachment pattern"
  - "Two-commit approach: first commit (narrow table) discovered to fail; second commit (wide table) is the correct fix"

metrics:
  duration: "~10 minutes"
  completed_date: "2026-03-30"
  tasks_completed: 2
  files_changed: 2
---

# Phase 01 Plan 05: Fix `_normalize_native_output` for Polars-Native Check Returns Summary

Fix `_normalize_native_output` in `checks.py` to wrap `pl.Series` and `pl.DataFrame` returns from `native=True` checks into narwhals, and add regression tests.

## What Was Built

**Task 1: Fix `_normalize_native_output`**

Added a polars handling block in `_normalize_native_output` (after the ibis block, before `return out`):

```python
try:
    import polars as pl
    if isinstance(out, pl.Series) or isinstance(out, pl.DataFrame):
        # Wide-table approach: attach boolean mask as CHECK_OUTPUT_KEY column on original frame
        # postprocess_lazyframe_output expects the original columns + CHECK_OUTPUT_KEY
        native_frame = nw.to_native(check_obj.frame)
        if isinstance(native_frame, pl.LazyFrame):
            native_frame = native_frame.collect()
        if isinstance(out, pl.Series):
            result = native_frame.with_columns(out.alias(CHECK_OUTPUT_KEY))
        else:  # pl.DataFrame — must have CHECK_OUTPUT_KEY column
            result = native_frame.with_columns(out[CHECK_OUTPUT_KEY].alias(CHECK_OUTPUT_KEY))
        return nw.from_native(result, eager_only=True)
except ImportError:
    pass
```

**Why wide-table**: `postprocess_lazyframe_output` calls `check_output.filter(~nw.col(CHECK_OUTPUT_KEY))` to get failure cases — it needs the original columns alongside the boolean mask to construct failure case metadata. A narrow single-column table (`to_frame(CHECK_OUTPUT_KEY)` alone) caused `ColumnNotFoundError` in downstream processing. The fix attaches the boolean output to the collected original frame.

**Task 2: Regression tests (`TestCustomChecksPolarsRowLevel`)**

Added `TestCustomChecksPolarsRowLevel` class to `tests/backends/narwhals/test_e2e.py` with 4 tests:
1. `test_native_series_check_passes` — `pl.Series` return, all rows pass
2. `test_native_series_check_fails` — `pl.Series` return, some rows fail → `SchemaError` raised
3. `test_native_dataframe_check_passes` — `pl.DataFrame` return, all rows pass
4. `test_native_dataframe_check_fails` — `pl.DataFrame` return, some rows fail → `SchemaError` raised

## Decisions Made

1. **Wide-table over narrow-table**: The initial fix used `out.to_frame(CHECK_OUTPUT_KEY)` (narrow), which passed the `postprocess()` type check but then failed with `ColumnNotFoundError` inside `postprocess_lazyframe_output`. The correct approach is to attach the output to the original collected frame.
2. **Collect before `with_columns`**: `pl.Series` is eager; attaching it to a `pl.LazyFrame` via `.with_columns()` requires collecting first.
3. **Two commits**: `cb39318c` was the initial (incorrect) narrow-table approach; `554c249b` is the corrected wide-table approach. The final state is correct.

## Deviations from Plan

None for requirements. Implementation approach for `pl.Series` is wide-table (attach to original frame) rather than the narrow `to_frame()` approach suggested in the plan context — the plan context contained a simpler suggestion that turned out to be incorrect. Agent self-corrected in `554c249b`.

## Verification Results

1. `python -m pytest tests/backends/narwhals/test_e2e.py -k "RowLevel" -v` — 4 tests pass ✓
2. `grep -n "isinstance(out, pl.Series)" pandera/backends/narwhals/checks.py` — line 113 ✓
3. Full narwhals test suite: 225 passed, 8 skipped, 1 xfailed ✓

## Self-Check: PASSED

- [x] `pandera/backends/narwhals/checks.py` — `_normalize_native_output` handles `pl.Series` and `pl.DataFrame`
- [x] `tests/backends/narwhals/test_e2e.py` — `TestCustomChecksPolarsRowLevel` with 4 passing tests
- [x] Commit `cb39318c` — fix(01-05): handle pl.Series and pl.DataFrame in _normalize_native_output (initial)
- [x] Commit `554c249b` — fix(01-05): wide-table approach for pl.Series and pl.DataFrame in native checks (corrected)
- [x] 225 narwhals tests pass
