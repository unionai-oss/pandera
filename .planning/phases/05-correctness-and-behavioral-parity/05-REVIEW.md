---
phase: 05-correctness-and-behavioral-parity
reviewed: 2026-05-25T00:00:00Z
depth: standard
files_reviewed: 5
files_reviewed_list:
  - pandera/backends/narwhals/container.py
  - tests/narwhals/test_phase01_arch.py
  - tests/pyspark/test_pyspark_accessor.py
  - tests/pyspark/test_pyspark_config.py
  - tests/pyspark/test_pyspark_model.py
findings:
  critical: 3
  warning: 4
  info: 2
  total: 9
status: issues_found
---

# Phase 05: Code Review Report

**Reviewed:** 2026-05-25T00:00:00Z
**Depth:** standard
**Files Reviewed:** 5
**Status:** issues_found

## Summary

Five files reviewed: the Narwhals container backend (`container.py`) and four test modules. The production code in `container.py` contains two logic bugs in `strict_filter_columns` (ordered column checking) and `check_column_values_are_unique` (empty-subset guard), plus a behavioral gap in the PySpark `drop_invalid_rows` path. Test files have reliability issues from conflicting `xfail` markers and module-level `SparkSession` creation that bypasses the session-scoped fixture. Unused imports in container.py are minor but clean-up worthy.

---

## Critical Issues

### CR-01: `UnboundLocalError` or silent mis-ordering in `strict_filter_columns` ordered check

**File:** `pandera/backends/narwhals/container.py:541-554`

**Issue:** When `schema.ordered=True`, `next_ordered_col` is assigned inside the `try` block but referenced unconditionally outside it (line 546). If the iterator is exhausted on the first schema column (`StopIteration` fires before any successful `next()`), `next_ordered_col` is never set, causing `UnboundLocalError`. If `StopIteration` fires after at least one successful `next()`, `next_ordered_col` retains the stale value from the previous iteration — the subsequent `if next_ordered_col != column` comparison uses the wrong expected value instead of raising the ordering error. The `StopIteration` branch should itself raise (as the pandas reference backend does), not silently fall through.

Compare with the pandas reference backend (`pandera/backends/pandas/container.py:567-581`): the `except StopIteration` branch appends a `SchemaError`, and the `if next_ordered_col != column` check is in the `else` branch so it only runs when a value was actually retrieved.

**Fix:**
```python
if schema.ordered and is_schema_col:
    try:
        next_ordered_col = next(sorted_column_names)
    except StopIteration:
        raise SchemaError(
            schema=schema,
            data=check_obj,
            message=f"column '{column}' out-of-order",
            failure_cases=column,
            check="column_ordered",
            reason_code=SchemaErrorReason.COLUMN_NOT_ORDERED,
        )
    else:
        if next_ordered_col != column:
            raise SchemaError(
                schema=schema,
                data=check_obj,
                message=f"column '{column}' out-of-order",
                failure_cases=column,
                check="column_ordered",
                reason_code=SchemaErrorReason.COLUMN_NOT_ORDERED,
            )
```

---

### CR-02: Missing empty-`subset` guard in `check_column_values_are_unique`

**File:** `pandera/backends/narwhals/container.py:627-633`

**Issue:** When `schema.unique` lists column names that do not exist in the frame, `subset` at line 628 becomes an empty list. The code then calls `check_obj.select([])` followed by `.group_by()` with no arguments and `.agg(nw.len().alias("_count"))`. Narwhals will raise an error (e.g., `InvalidOperationError` or similar) rather than cleanly returning a "no duplicates" result. The pandas backend guards this with `if not subset: continue` (see `pandera/backends/pandas/container.py:877-878`). This also makes `test_dataframe_schema_unique_wrong_column` rely entirely on the `xfail` marker rather than reaching any assertion.

**Fix:**
```python
for lst in temp_unique:
    subset = [x for x in lst if x in frame_column_names]
    if not subset:          # ← add this guard (matches pandas backend)
        continue
    grouped = (
        check_obj.select(subset)
        ...
    )
```

---

### CR-03: PySpark `drop_invalid_rows` path skips accessor setup

**File:** `pandera/backends/narwhals/container.py:225-230`

**Issue:** When `drop_invalid_rows=True` and the backend is PySpark, the code takes the `drop_invalid_rows` branch (lines 225-230) and returns `check_obj_parsed` directly — without calling `_handle_pyspark_validation_result`. As a result, `check_obj.pandera.schema` is never set and `check_obj.pandera.errors` is never initialized. Any downstream code that reads `.pandera.errors` on the returned PySpark DataFrame will raise `AttributeError` (or get a stale value from a prior validation run if the object was reused).

The normal PySpark success/error paths (lines 231-245) always route through `_handle_pyspark_validation_result`, which sets both `.pandera.schema` and `.pandera.errors`. The `drop_invalid_rows` path must do the same.

**Fix:**
```python
if getattr(schema, "drop_invalid_rows", False):
    check_obj_parsed = _to_frame_kind_nw(check_lf, return_type)
    check_obj_parsed = self.drop_invalid_rows(
        check_obj_parsed, error_handler
    )
    if is_pyspark:
        return self._handle_pyspark_validation_result(
            check_obj_parsed, error_handler, schema, has_errors=True
        )
    return check_obj_parsed
```

---

## Warnings

### WR-01: `assert all(check_passed)` in production code bypassed under `python -O`

**File:** `pandera/backends/narwhals/container.py:367`

**Issue:** `run_schema_component_checks` uses a bare `assert all(check_passed)` to verify that all successful validations returned non-None results. Python's optimized mode (`python -O`) strips all `assert` statements — the invariant is silently not checked. Additionally, `AssertionError` is not a user-friendly error; if the assertion ever fired it would produce an opaque traceback. Replace with an explicit runtime check.

**Fix:**
```python
# Replace:
assert all(check_passed)
# With (or remove entirely if check_passed is always True by construction):
if not all(check_passed):
    raise SchemaDefinitionError(
        "One or more schema component validations returned None unexpectedly."
    )
```
If `result is not None` is always true when no exception is raised (which appears to be the case), the `check_passed` list and assertion can be removed altogether.

---

### WR-02: Conflicting `@pytest.mark.xfail` markers on `test_registered_dataframemodel_checks`

**File:** `tests/pyspark/test_pyspark_model.py:550-555`

**Issue:** Two `@pytest.mark.xfail` decorators are stacked on the same test. The outer marker is `@pytest.mark.xfail(raises=ValueError)` (no `strict`), and the inner is `@pytest.mark.xfail(condition=CONFIG.use_narwhals_backend, ..., strict=True)`. Pytest collects both markers and evaluates both conditions, but when both are active simultaneously the non-strict outer marker can mask the `strict=True` inner marker. If the narwhals-backend run unexpectedly passes, the outer non-strict `xfail(raises=ValueError)` could satisfy the "test xfailed but passed" condition without triggering the `strict=True` failure. The intent (two independent failure modes) should be expressed more carefully.

**Fix:** Separate the two failure modes explicitly or use `pytest.mark.xfail(condition=..., reason=..., raises=(ValueError, SomeOtherError), strict=True)` combined into a single marker with a clear `raises` tuple.

---

### WR-03: Module-level `SparkSession` creation in `test_pyspark_accessor.py`

**File:** `tests/pyspark/test_pyspark_accessor.py:14-15`

**Issue:** `spark = SparkSession.builder.getOrCreate()` is called at module import time (line 14), not through the session-scoped `spark` fixture in `conftest.py`. This means:

1. The `conftest.py` fixture's teardown (`spark.stop()`) may conflict with the module-level session.
2. The PySpark configuration set in `conftest.py` (e.g., Hadoop/warehouse dir workarounds for PySpark 4.0+) is not applied.
3. Test parametrization over `spark`/`spark_connect` sessions (used in all other pyspark test modules via `pytestmark`) is absent — `test_dataframe_add_schema` only runs against a single ad-hoc session.

**Fix:** Remove the module-level `spark = ...` lines and add `spark_session` fixture parametrization or use `request.getfixturevalue("spark")` inside the test, matching the pattern in `test_pyspark_model.py` and `test_pyspark_config.py`.

---

### WR-04: `check_column_presence` error message forces eager materialization on lazy/SQL backends

**File:** `pandera/backends/narwhals/container.py:595`

**Issue:** `_to_native(check_obj.head())` is called when building the error message string for a missing-column error. For SQL-lazy backends (PySpark, ibis), calling `.head()` on the narwhals LazyFrame triggers query execution (a network or cluster round-trip). For polars LazyFrames, `str(pl.LazyFrame)` returns a non-informative plan representation rather than actual data. In both cases the diagnostic value is low and the cost can be high.

**Fix:**
```python
# Replace:
f"column '{colname}' not in dataframe\n{_to_native(check_obj.head())}"
# With something that doesn't trigger execution:
f"column '{colname}' not in dataframe. Available columns: {check_obj.collect_schema().names()}"
```

---

## Info

### IN-01: Unused imports in `container.py`

**File:** `pandera/backends/narwhals/container.py:10,42`

**Issue:** `Optional` (from `typing`) and `validation_type` (from `pandera.validation_depth`) are imported but never referenced in the module body.

**Fix:** Remove the two unused names:
```python
# line 10: remove Optional
from typing import TYPE_CHECKING, Any
# line 42: remove validation_type
from pandera.validation_depth import validate_scope
```

---

### IN-02: `test_modin_accessor_warning` misleading function name

**File:** `tests/pyspark/test_pyspark_accessor.py:64`

**Issue:** The test is named `test_modin_accessor_warning` but tests the PySpark accessor (`pyspark_sql_accessor.register_dataframe_accessor`). The name appears to be a copy-paste artifact from the modin test suite.

**Fix:** Rename to `test_pyspark_accessor_warning` for clarity.

---

_Reviewed: 2026-05-25T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
