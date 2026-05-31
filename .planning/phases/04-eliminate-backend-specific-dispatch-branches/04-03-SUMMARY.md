---
phase: 04-eliminate-backend-specific-dispatch-branches
plan: "03"
subsystem: narwhals-backend
tags:
  - narwhals
  - pyspark
  - refactoring
  - arch-03
dependency_graph:
  requires:
    - "04-RESEARCH.md (ARCH-03 analysis)"
  provides:
    - "Schema-driven check_dtype dispatch (uses_pyspark_dtype)"
  affects:
    - "pandera/backends/narwhals/components.py"
    - "tests/pyspark/test_pyspark_dtypes.py"
    - "tests/narwhals/test_arch03_schema_driven_dispatch.py"
tech_stack:
  added: []
  patterns:
    - "isinstance(schema.dtype, pyspark_engine.DataType) probe replaces check_obj.implementation probe"
    - "Lazy import of pyspark_engine inside check_dtype method (same pattern as narwhals_engine)"
key_files:
  created:
    - "tests/narwhals/test_arch03_schema_driven_dispatch.py"
  modified:
    - "pandera/backends/narwhals/components.py"
    - "tests/pyspark/test_pyspark_dtypes.py"
decisions:
  - "Schema-driven dispatch: isinstance(schema.dtype, pyspark_engine.DataType) replaces frame-based check_obj.implementation in (PYSPARK, PYSPARK_CONNECT)"
  - "pyspark_engine imported lazily inside check_dtype to avoid circular imports, matching existing narwhals_engine pattern"
  - "str(pyspark_dtype) == str(schema.dtype) comparison logic stays unchanged — only the detection mechanism changes"
metrics:
  duration: "~3 minutes"
  completed: "2026-05-25"
  tasks_completed: 2
  files_changed: 3
---

# Phase 04 Plan 03: Schema-Driven check_dtype Dispatch Summary

**One-liner:** Replace frame-implementation probe with `isinstance(schema.dtype, pyspark_engine.DataType)` schema-driven dispatch in `check_dtype`, satisfying ARCH-03.

## What Was Built

### Task 1: Replace frame-implementation probe (TDD)
Modified `check_dtype` in `pandera/backends/narwhals/components.py` to replace:
```python
is_pyspark = check_obj.implementation in (
    nw.Implementation.PYSPARK,
    nw.Implementation.PYSPARK_CONNECT,
)
```
with:
```python
from pandera.engines import pyspark_engine as _pyspark_engine
uses_pyspark_dtype = isinstance(schema.dtype, _pyspark_engine.DataType)
```

The `str(pyspark_dtype) == str(schema.dtype)` comparison logic itself is unchanged — only the detection mechanism changed from frame-driven to schema-driven. Updated inline comments explain the rationale with reference to ARCH-03.

TDD cycle followed:
- RED commit `3b778b0c`: 4 structural tests asserting schema-driven probe — all failed before implementation
- GREEN commit `cbc25876`: Implementation makes all 5 tests pass

### Task 2: Add explanatory comment to verifySchema=False workaround
Added a multi-line comment above the `if CONFIG.use_narwhals_backend:` block in `tests/pyspark/test_pyspark_dtypes.py` explaining:
- WHY an empty single-column DataFrame is created (avoids STRUCT_ARRAY_LENGTH_MISMATCH)
- Root cause: `conftest.spark_df()` uses `verifySchema=False` with multi-value rows against a single-column schema
- Trigger: narwhals backend calls `.first()` inside `_materialize()` which triggers Spark row structure validation
- Scope: test-fixture correction, not a backend workaround; fixing conftest is out of scope for ARCH-03

## Deviations from Plan

None — plan executed exactly as written.

## Verification Results

### Source assertions (all pass):
- `grep -n "is_pyspark" pandera/backends/narwhals/components.py` — no matches
- `grep -n "uses_pyspark_dtype" pandera/backends/narwhals/components.py` — 3 matches (definition + 2 call sites)
- `grep -n "isinstance.*_pyspark_engine\.DataType" pandera/backends/narwhals/components.py` — 1 match
- `grep -n "check_obj.implementation in" pandera/backends/narwhals/components.py` — no matches

### Test results:
- `tests/narwhals/test_arch03_schema_driven_dispatch.py` — 5/5 passed (GREEN)
- `tests/polars/ tests/ibis/` — 502 passed, 1 skipped, 134 xfailed (no regressions)

## TDD Gate Compliance

| Gate | Commit | Status |
|------|--------|--------|
| RED (test commit) | 3b778b0c | 4 failing structural tests written first |
| GREEN (feat commit) | cbc25876 | All 5 tests pass after implementation |
| REFACTOR | N/A | No refactoring needed |

## Self-Check: PASSED

Files created/modified:
- `tests/narwhals/test_arch03_schema_driven_dispatch.py` — FOUND (new)
- `pandera/backends/narwhals/components.py` — FOUND (modified)
- `tests/pyspark/test_pyspark_dtypes.py` — FOUND (modified)

Commits:
- `3b778b0c` — test(04-03): RED tests — FOUND
- `cbc25876` — feat(04-03): GREEN implementation — FOUND
- `34e91c19` — docs(04-03): Task 2 comment — FOUND
