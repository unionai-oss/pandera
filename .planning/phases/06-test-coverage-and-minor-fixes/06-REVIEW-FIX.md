---
phase: 06-test-coverage-and-minor-fixes
fixed_at: 2026-05-26T00:17:00Z
review_path: .planning/phases/06-test-coverage-and-minor-fixes/06-REVIEW.md
iteration: 1
findings_in_scope: 7
fixed: 7
skipped: 0
status: all_fixed
---

# Phase 06: Code Review Fix Report

**Fixed at:** 2026-05-26T00:17:00Z
**Source review:** .planning/phases/06-test-coverage-and-minor-fixes/06-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 7
- Fixed: 7
- Skipped: 0

## Fixed Issues

### CR-01: `strict_filter_columns` — stale / unbound `next_ordered_col` after `StopIteration`

**Files modified:** `pandera/backends/narwhals/container.py`, `pandera/backends/ibis/container.py`
**Commit:** cd464937
**Applied fix:** Replaced `try/except StopIteration: pass` followed by unconditional `if next_ordered_col != column` with the correct `try/except/else` pattern (mirroring the pandas reference backend). The `StopIteration` branch now raises `SchemaError` directly (iterator exhausted means out-of-order); the `else` branch runs the comparison only when iteration succeeded. Applied identically to both backends.

---

### CR-02: `run_schema_component_checks` — `assert all(check_passed)` is an incorrect production guard

**Files modified:** `pandera/backends/narwhals/container.py`, `pandera/backends/ibis/container.py`
**Commit:** c0a3fc4b
**Applied fix:** Removed the `check_passed` list and the `assert all(check_passed)` statement from both backends. The validate call is no longer assigned to `result`; the absence of an exception is the success signal. The `assert` was both silently stripped by `-O` builds and vacuously true when all components raised exceptions (empty list).

---

### WR-01: Missing `assert` in `test_strict_filter` — the "filter" branch is never actually verified

**Files modified:** `tests/ibis/test_ibis_container.py`
**Commit:** 17a06937
**Applied fix:** Added `assert` before `filtered_data.execute().equals(t_basic.execute())` so the boolean result is actually checked rather than silently discarded.

---

### WR-02: Dead code in `test_drop_invalid_rows` — `got` and `expected` computed but not used

**Files modified:** `tests/ibis/test_ibis_container.py`
**Commit:** 9fad2c3d
**Applied fix:** Changed `assert validated_data.execute().equals(expected_valid_data.execute())` to `assert got.equals(expected)`, using the already-computed variables. This eliminates the two redundant `.execute()` SQL round-trips.

---

### WR-03: Duplicate parametrize value in `test_different_unique_settings`

**Files modified:** `tests/ibis/test_ibis_container.py`
**Commit:** af1e742e
**Applied fix:** Removed the duplicate `("exclude_first", [4, 5, 6, 7])` entry from the parametrize list. All three unique settings (`exclude_first`, `all`, `exclude_last`) remain covered by the remaining three entries; the redundant identical test case is eliminated.

---

### WR-04: `_spark_env_vars` autouse fixture sets environment variables without cleanup

**Files modified:** `tests/narwhals/test_e2e.py`
**Commit:** d45689d1
**Applied fix:** Rewrote the fixture to save prior values of `SPARK_LOCAL_IP` and `PYARROW_IGNORE_TIMEZONE` before setting them, then restore (or delete if previously absent) in teardown after `yield`. Early return when `HAS_PYSPARK` is false avoids the yield entirely for non-PySpark environments.

---

### WR-05: `TestCustomChecksIbis.test_schema_level_check_passes` — name and docstring contradict the test body

**Files modified:** `tests/narwhals/test_e2e.py`
**Commit:** 311f29c7
**Applied fix:** Renamed method to `test_schema_level_check_fails_on_invalid_data` and updated docstring to `"Schema-level ibis check (key='*') raises SchemaError when check condition fails."` The test body (which expects `SchemaError` with data where `x < y` everywhere) now accurately matches the test name.

---

_Fixed: 2026-05-26T00:17:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
