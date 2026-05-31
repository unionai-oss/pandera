---
phase: 01-structural-cleanup
plan: "01"
subsystem: narwhals-utils
tags: [lazy-detection, import-cleanup, types-01, types-02, clean-04]
requirements: [TYPES-01, TYPES-02]

dependency_graph:
  requires: []
  provides: [_is_lazy-utility]
  affects:
    - pandera/api/narwhals/utils.py
    - pandera/backends/narwhals/base.py
    - pandera/backends/narwhals/container.py
    - pandera/backends/narwhals/components.py

tech_stack:
  added: []
  patterns:
    - "_is_lazy(frame) as single canonical lazy-detection function in utils.py"
    - "from pandera.api.narwhals.utils import _is_lazy in all consuming modules"

key_files:
  created: []
  modified:
    - pandera/api/narwhals/utils.py
    - pandera/backends/narwhals/base.py
    - pandera/backends/narwhals/container.py
    - pandera/backends/narwhals/components.py

decisions:
  - "_is_lazy mirrors _is_lazy_or_sql body exactly but lives in utils.py — single canonical location"
  - "container.py and components.py import _is_lazy but still use inline hasattr check — import added but call site not updated"

metrics:
  duration: "~7 minutes"
  completed_date: "2026-03-30"
  tasks_completed: 1
  files_changed: 4
---

# Phase 01 Plan 01: Foundation — `_is_lazy` Utility Summary

Create the unified `_is_lazy` utility in `utils.py` and remove the scattered `_is_lazy_or_sql` from `base.py`.

## What Was Built

**Task 1: Create `_is_lazy` utility and replace scattered lazy detection (COMPLETE)**

- Added `_is_lazy(frame) -> bool` to `pandera/api/narwhals/utils.py` — canonical lazy/SQL-lazy detection function
- Removed `_is_lazy_or_sql` from `pandera/backends/narwhals/base.py`
- Updated import in `base.py`: `from pandera.api.narwhals.utils import _is_lazy, _materialize`
- Replaced `_is_lazy_or_sql(fc)` call in `failure_cases_metadata` with `_is_lazy(fc)`
- Added `_is_lazy` import to `container.py` and `components.py`

**Task 2: Hoist inner imports and add TYPE_CHECKING guard (NOT EXECUTED)**

Agent ran out of context before executing Task 2. The following items remain open (CLEAN-04):
- `import re` inside `container.py:check_column_presence` (line ~448) — not hoisted
- `import re` inside `components.py:get_regex_columns` (line ~102) — not hoisted
- `from pandera.api.polars.container import DataFrameSchema` at runtime (line 14) — no TYPE_CHECKING guard added
- `narwhals_engine.py` inner imports not hoisted

## Decisions Made

1. `_is_lazy` body is identical to `_is_lazy_or_sql` — rename only, no behavioral change.
2. `_is_lazy` import added to `container.py` and `components.py` but those files still use inline `hasattr(nw.to_native(fc), "execute")` at the affected call sites. The import is present but the call sites were not updated to use `_is_lazy(fc)`.

## Deviations from Plan

**Task 2 not executed.** Agent hit usage limit after Task 1. CLEAN-04 requirements (import hoisting, TYPE_CHECKING guard for polars DataFrameSchema) remain open. These are structural cleanup items — no functional impact.

TYPES-01 and TYPES-02 are satisfied. CLEAN-04 is not satisfied by this plan.

## Verification Results

1. `grep -n "def _is_lazy" pandera/api/narwhals/utils.py` — 1 match at line 34 ✓
2. `grep -n "_is_lazy_or_sql" pandera/backends/narwhals/` — zero matches ✓
3. Narwhals test suite passes (verified by 01-02 agent: 221 passed, 8 skipped, 1 xfailed)

## Self-Check: PASSED (partial)

- [x] `pandera/api/narwhals/utils.py` — contains `def _is_lazy`
- [x] `pandera/backends/narwhals/base.py` — `_is_lazy_or_sql` removed, `_is_lazy` imported and used
- [x] Commit `860c5a1a` — feat(01-01): create _is_lazy utility and replace scattered lazy detection
- [ ] TYPE_CHECKING guard in container.py — NOT DONE (CLEAN-04 gap)
- [ ] import re hoisted in container.py and components.py — NOT DONE (CLEAN-04 gap)
- [ ] narwhals_engine.py inner imports hoisted — NOT DONE (CLEAN-04 gap)
