---
phase: 01-structural-cleanup
plan: "02"
subsystem: narwhals-engine
tags: [eager-execution, materialization, coerce, ibis, polars]
requirements: [EAGER-01, EAGER-02]

dependency_graph:
  requires: []
  provides: [bounded-try-coerce-probe, materialization-audit]
  affects:
    - pandera/engines/narwhals_engine.py
    - pandera/backends/narwhals/container.py
    - pandera/backends/narwhals/components.py

tech_stack:
  added: []
  patterns:
    - "head(1).collect() bounded probe instead of full-frame collect() in try_coerce"
    - "_materialize() for ibis-safe DataFrame probe"
    - "Explanatory comments on every materialization call"

key_files:
  created: []
  modified:
    - pandera/engines/narwhals_engine.py
    - pandera/backends/narwhals/container.py
    - pandera/backends/narwhals/components.py

decisions:
  - "Use head(1).collect() for polars LazyFrame probe and _materialize(head(1)) for ibis DataFrame — differentiates by isinstance(lf, nw.LazyFrame)"
  - "Hoist _materialize import to module level in narwhals_engine.py — always-available pandera code, safe to hoist"
  - "Error handler path uses _to_native(_materialize(frame)) instead of frame.collect() — handles ibis backends that don't support .collect()"

metrics:
  duration: "~10 minutes"
  completed_date: "2026-03-30"
  tasks_completed: 2
  files_changed: 3
---

# Phase 01 Plan 02: Eliminate Unnecessary Eager Execution Summary

Replace full-frame `lf.collect()` in `try_coerce` with a bounded `head(1)` probe, and audit `container.py`/`components.py` for unbounded materializations.

## What Was Built

**Task 1: Bounded head(1) probe in try_coerce**

Replaced the unbounded `lf.collect()` in `DataType.try_coerce()` with a 1-row probe:
- For `nw.LazyFrame` (polars): `lf.head(1).collect()` — stays in narwhals, exercises the cast path
- For `nw.DataFrame` (ibis): `_materialize(lf.head(1))` — handles `.execute()` via `_materialize`
- Error handler path: replaced `data_container.frame.collect()` with `_to_native(_materialize(data_container.frame))` to handle ibis backends that don't support `.collect()`
- Added `from pandera.api.narwhals.utils import _materialize` at module level in `narwhals_engine.py`

**Task 2: Materialization audit of container.py and components.py**

Audited all `_materialize()` and `.collect()` calls. Findings:

| File | Location | Call | Classification |
|------|----------|------|----------------|
| container.py | `_to_frame_kind_nw` | `native.collect()` | Final return boundary — caller-requested materialization |
| container.py | `validate` error path | `_materialize(fc)` | Error path — bounded to failing rows only |
| container.py | `check_column_values_are_unique` | `_materialize(dup_rows)` | Bounded — only duplicate rows after group_by+filter |
| components.py | `check_nullable` | `_materialize(combined_lf.select(.any()))` | Single-row scalar — 1 boolean value |
| components.py | `check_unique` | `_materialize(dup_values)` | Bounded — only duplicate column values |
| components.py | `run_checks_and_handle_errors` error path | `_materialize(fc)` | Error path — bounded to failing rows only |

**Conclusion**: No unbounded hot-path materializations exist in container.py or components.py. All calls are either bounded (duplicate rows/values only), single-row scalars, error paths (failure_cases), or the final return-type conversion at schema exit.

All calls now have explanatory comments documenting why they are acceptable.

## Decisions Made

1. **head(1) probe approach**: Uses `isinstance(lf, nw.LazyFrame)` to differentiate polars (uses `.collect()`) from ibis (uses `_materialize()` which calls `.execute()`). This is cleaner than adding a `_materialize` wrapper that also handles `.collect()` since the narwhals type distinction already captures the semantic difference.

2. **_materialize hoisted to module level**: Previously `_materialize` was only imported in method scope. Moving it to module level in `narwhals_engine.py` avoids redundant inner imports and is safe since `pandera.api.narwhals` is always-available pandera code (not an optional dependency).

3. **Error handler path fix**: The original `data_container.frame.collect()` would raise `AttributeError` for ibis frames (which don't support `.collect()` on the native object). Replacing with `_to_native(_materialize(data_container.frame))` handles both backends correctly.

## Deviations from Plan

None — plan executed exactly as written.

## Verification Results

1. `python -m pytest tests/backends/narwhals/ -q` — **221 passed, 8 skipped, 1 xfailed** (all green)
2. `grep -n "\.collect()" pandera/engines/narwhals_engine.py` — only `lf.head(1).collect()` pattern, no bare `.collect()`
3. All `_materialize()` and `.collect()` calls in container.py and components.py have explanatory comments

## Self-Check: PASSED

- [x] `pandera/engines/narwhals_engine.py` — modified, contains `lf.head(1).collect()`
- [x] `pandera/backends/narwhals/container.py` — modified, all calls commented
- [x] `pandera/backends/narwhals/components.py` — modified, all calls commented
- [x] Commit `a2528262` — feat(01-02): replace full-frame collect in try_coerce with bounded head(1) probe
- [x] Commit `eb6783b0` — chore(01-02): audit container.py and components.py materialization calls
- [x] 221 narwhals tests pass
