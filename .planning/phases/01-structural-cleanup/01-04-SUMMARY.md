---
phase: 01-structural-cleanup
plan: "04"
subsystem: narwhals-container
tags: [infer-columns, importlib, arch-tests, clean-01, clean-02]
requirements: [CLEAN-01, CLEAN-02]

dependency_graph:
  requires: []
  provides: [infer-columns-method, clean-01-arch-test, clean-02-arch-test]
  affects:
    - pandera/api/dataframe/container.py
    - pandera/backends/narwhals/container.py
    - pandera/backends/narwhals/checks.py
    - tests/backends/narwhals/test_phase01_arch.py

tech_stack:
  added: []
  patterns:
    - "schema.infer_columns(column_names) as the Column class lookup mechanism"
    - "Source-inspection arch tests using inspect.getsource()"

key_files:
  created:
    - tests/backends/narwhals/test_phase01_arch.py (pre-existing, extended with 4 new tests)
  modified:
    - pandera/api/dataframe/container.py
    - pandera/backends/narwhals/container.py
    - pandera/backends/narwhals/checks.py

decisions:
  - "infer_columns() uses type(next(iter(self.columns.values()))) when columns exist — reliable, no importlib needed"
  - "infer_columns() falls back to importlib when schema has no columns — same semantics as old pattern, now encapsulated"
  - "01-04 also included prerequisite commit from origin/main: Unify builtin comparison checks with Narwhals impl (adds CheckData protocol, frame property to PolarsData/IbisData, narwhals-based equal_to/not_equal_to etc. in builtin_checks.py)"
  - "CLEAN-01 violation found in checks.py (import polars in postprocess_bool_output) — fixed inline as part of arch test task; uses nw.from_dict(backend='pyarrow').lazy() instead"

metrics:
  duration: "~10 minutes"
  completed_date: "2026-03-30"
  tasks_completed: 2
  files_changed: 4
---

# Phase 01 Plan 04: `infer_columns()` + Arch Tests Summary

Add `infer_columns()` to the generic `DataFrameSchema` base class and wire `collect_schema_components` to use it, eliminating the `importlib` dynamic Column class lookup. Add CLEAN-01 and CLEAN-02 arch tests.

## What Was Built

**Task 1: Add `infer_columns()` and wire container.py**

- Added `infer_columns(self, column_names: list) -> list` to `pandera/api/dataframe/container.py` generic `DataFrameSchema` base
  - When `self.columns` is non-empty: infers Column class from existing column entries via `type(next(iter(self.columns.values())))`
  - When `self.columns` is empty: falls back to importlib lookup (same semantics as old pattern, now encapsulated in schema layer)
- Updated `collect_schema_components` in `pandera/backends/narwhals/container.py`:
  - Removed `import importlib`, `_pkg = schema.__class__.__module__...`, `Column = importlib.import_module(...)` 3-line block
  - Replaced with `schema.infer_columns(frame_column_names)` — narwhals backend is now fully framework-agnostic for Column construction
  - Removed TODO comment that tracked this cleanup

**Task 2: Add CLEAN-01 and CLEAN-02 arch tests**

Added 4 tests to `tests/backends/narwhals/test_phase01_arch.py` (file now has 17 total tests):
1. `test_checks_has_no_polars_import` — source-inspects `checks.py` for `import polars`
2. `test_container_has_no_polars_components_import` — source-inspects `container.py` for `pandera.api.polars.components` and `importlib.import_module`
3. `test_container_uses_infer_columns_for_schema_components` — source-inspects `collect_schema_components` for `schema.infer_columns(`
4. `test_infer_columns_returns_correct_column_type_for_polars` — integration test: creates `pa_pl.DataFrameSchema(dtype=pl.Int64)`, calls `infer_columns(["a","b"])`, asserts results are `PolarsColumn` instances

**Also included: prerequisite from `origin/main`**

Cherry-picked `Unify builtin comparison checks with Narwhals impl` (originally from `origin/refactor/narwhalify`, already merged to `origin/main`):
- Adds `CheckData` protocol to `pandera/api/base/types.py`
- Adds `.frame` property to `PolarsData` and `IbisData` for uniform access
- Implements `equal_to`, `not_equal_to`, `greater_than`, `greater_than_or_equal_to`, `less_than`, `less_than_or_equal_to` in `backends/base/builtin_checks.py` using narwhals
- Polars backend delegates to base implementations

**CLEAN-01 hotfix**

While adding the arch test `test_checks_has_no_polars_import`, the test found a real violation in `checks.py:postprocess_bool_output` which imported `polars` for a fallback eager frame construction. Fixed by using `nw.from_dict({"x": [1]}, backend='pyarrow').lazy()` as ibis-fallback — no polars dependency in `checks.py` now.

## Decisions Made

1. `infer_columns()` placed after `__init__` and before `_validate_attributes` in `DataFrameSchema` — alongside other schema utility methods.
2. The importlib fallback inside `infer_columns()` is intentional — when no columns exist, the schema can't infer the Column class from existing entries. This matches prior semantics.
3. CLEAN-01 violation fixed immediately rather than deferred — arch test + fix in same commit.

## Deviations from Plan

None for the stated requirements. One addition: the prerequisite `builtin_checks.py` unification commit was cherry-picked from origin/main (date Jan 19, 2026) as it was needed to avoid test failures with the narwhals-based check dispatch.

## Verification Results

1. `grep -n "def infer_columns" pandera/api/dataframe/container.py` — line 190 ✓
2. `grep -n "importlib" pandera/backends/narwhals/container.py` — zero matches ✓
3. `grep -n "polars.components" pandera/backends/narwhals/container.py` — zero matches ✓
4. All 225 narwhals tests pass (221 original + 4 new arch tests)

## Self-Check: PASSED

- [x] `pandera/api/dataframe/container.py` — `def infer_columns` at line 190
- [x] `pandera/backends/narwhals/container.py` — `schema.infer_columns(` at line 346, no `importlib`
- [x] `tests/backends/narwhals/test_phase01_arch.py` — 4 new CLEAN-01/CLEAN-02 tests, all pass
- [x] Commit `39f779c2` — Unify builtin comparison checks with Narwhals impl
- [x] Commit `75e11466` — feat(01-04): add infer_columns() to DataFrameSchema base, wire narwhals container
- [x] Commit `e19af983` — test(01-04): add CLEAN-01 and CLEAN-02 arch tests; fix polars import in checks.py
- [x] 225 narwhals tests pass
