---
plan: 02-02
phase: 02-remaining-pr-review-fixes
status: complete
completed: 2026-03-22
commits:
  - refactor(02-02): simplify check_dtype via cross-engine narwhals Engine.dtype()
---

## What Was Built

Refactored two methods in `ColumnBackend` (`components.py`) per PR #2223 reviewer feedback.

**Task 1 — check_nullable:** Replaced separate `_materialize` + horizontal concat with a single `check_obj.with_columns(null_expr.alias(CHECK_OUTPUT_KEY))` followed by a filter. No more separate frame materialization.

**Task 2 — check_dtype:** Dropped the three-pass polars/ibis native fallback ladder. Instead, extended `narwhals_engine.Engine.dtype()` to accept cross-engine dtypes (polars_engine, ibis_engine) by re-interpreting through the shared abstract pandera base class (e.g. `polars_engine.Int64` → `dtypes.Int64` → `narwhals_engine.Int64`). `check_dtype` now does a single narwhals-engine comparison. Parametric types (List, Struct) fall back to a direct check since `pandera.dtypes` has no abstract List/Struct.

## Key Files

- `pandera/backends/narwhals/components.py` — check_nullable and check_dtype refactored
- `pandera/engines/narwhals_engine.py` — Engine.dtype() extended for cross-engine inputs

## Deviations

The plan specified "single narwhals-engine pass with no polars_engine or ibis_engine fallback." The implementation achieves this goal but required extending `narwhals_engine.Engine.dtype()` (not mentioned in the plan) rather than embedding the conversion logic directly in `check_dtype`. This is the cleaner architectural location for that concern.

## Test Results

125 passed, 1 skipped, 3 xfailed, 4 xpassed — baseline unchanged.
