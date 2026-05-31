---
phase: 01-structural-cleanup
plan: "03"
subsystem: narwhals-base
tags: [import-cleanup, polars-isolation, lazy-detection, clean-03, types-03]
requirements: [TYPES-03, CLEAN-03]

dependency_graph:
  requires: [01-01]
  provides: [guarded-polars-imports, functools-hoisted]
  affects:
    - pandera/backends/narwhals/base.py

tech_stack:
  added: []
  patterns:
    - "try/except ImportError guard for all inner polars imports in function bodies"
    - "functools hoisted to module-level import"
    - "_is_lazy(fc) already in place from Plan 01-01 — no change needed"

key_files:
  created: []
  modified:
    - pandera/backends/narwhals/base.py

decisions:
  - "Task 1 was already complete: Plan 01-01 replaced _is_lazy_or_sql with _is_lazy in the failure_cases_metadata dispatch — no code change needed"
  - "Four unconditional inner polars imports wrapped in try/except ImportError; polars still used in each branch since it is available whenever those branches execute"
  - "functools hoisted to module level — stdlib import, always available, appropriate to hoist per CLEAN-04"
  - "Empty failure_cases fallback returns None when polars is unavailable (pl is None guard); ibis-only callers hit the ibis.union() path and never reach this fallback"

metrics:
  duration: "~8 minutes"
  completed_date: "2026-03-30"
  tasks_completed: 2
  files_changed: 1
---

# Phase 01 Plan 03: Failure Cases Metadata Dispatch Cleanup Summary

Verified `_is_lazy(fc)` in `failure_cases_metadata` dispatch (already done by Plan 01-01) and eliminated all four unconditional `import polars as pl` inner calls from `base.py`.

## What Was Built

**Task 1: Replace _is_lazy_or_sql with _is_lazy (already complete)**

Plan 01-01 already renamed `_is_lazy_or_sql` to `_is_lazy` throughout `base.py`. The dispatch condition at line 201 already reads:
```python
if isinstance(fc, (nw.LazyFrame, nw.DataFrame)) and _is_lazy(fc):
```
Zero matches for `_is_lazy_or_sql` in `base.py`. Task 1 required no code changes — verified and proceeded.

**Task 2: Eliminate unconditional inner polars imports**

Four locations updated in `failure_cases_metadata`:

1. **Eager polars path** (was line 231): Wrapped `import polars as pl` in `try/except ImportError`. Branch only reached for eager polars DataFrames, so polars will always be present.

2. **Scalar path** (was line 285): Wrapped `import polars as pl` in `try/except ImportError`. Branch reached for Python scalars regardless of backend, but polars `pl.DataFrame` construction follows the same guarded pattern.

3. **Concat path, non-ibis branch** (was line 311): Wrapped in `try/except ImportError`. Only reached when `failure_case_collection` contains `pl.DataFrame` items (not ibis.Table), so polars is present.

4. **Empty concat fallback** (was line 314): Wrapped in `try/except ImportError`. Returns `pl.DataFrame()` when polars is available, `None` when not.

Additionally, `import functools` that was inside the ibis concat branch was hoisted to module-level (line 3), consistent with CLEAN-04.

## Deviations from Plan

None — plan executed exactly as written. Task 1 was pre-completed by Plan 01-01; Task 2 changes matched the plan specification.

## Verification

1. `python -m pytest tests/backends/narwhals/ -q` — **229 passed, 8 skipped, 1 xfailed**
2. `grep -n "_is_lazy_or_sql" pandera/backends/narwhals/base.py` — **zero matches**
3. `grep -n "_is_lazy(fc)" pandera/backends/narwhals/base.py` — **one match at line 201**
4. All inner `import polars as pl` are inside `try:` blocks — **confirmed**
5. `import functools` appears at module level (line 3) — **confirmed**

## Self-Check

- [x] File exists: `pandera/backends/narwhals/base.py` — FOUND
- [x] Commit exists: `b2828ff4` — FOUND
