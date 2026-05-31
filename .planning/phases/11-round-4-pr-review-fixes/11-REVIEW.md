---
phase: 11-round-4-pr-review-fixes
reviewed: 2026-05-30T00:00:00Z
depth: standard
files_reviewed: 19
files_reviewed_list:
  - docs/source/pyspark_sql.md
  - docs/source/supported_libraries.md
  - pandera/api/narwhals/utils.py
  - pandera/backends/ibis/container.py
  - pandera/backends/narwhals/base.py
  - pandera/backends/narwhals/container.py
  - tests/ibis/test_ibis_container.py
  - tests/narwhals/test_e2e.py
  - tests/narwhals/test_phase01_arch.py
  - tests/narwhals/test_phase02_validate_helper.py
  - tests/pyspark/conftest.py
  - tests/pyspark/test_pyspark_accessor.py
  - tests/pyspark/test_pyspark_check.py
  - tests/pyspark/test_pyspark_config.py
  - tests/pyspark/test_pyspark_container.py
  - tests/pyspark/test_pyspark_decorators.py
  - tests/pyspark/test_pyspark_dtypes.py
  - tests/pyspark/test_pyspark_error.py
  - tests/pyspark/test_pyspark_model.py
findings:
  critical: 2
  warning: 4
  info: 2
  total: 8
status: issues_found
---

# Phase 11: Code Review Report

**Reviewed:** 2026-05-30
**Depth:** standard
**Files Reviewed:** 19
**Status:** issues_found

## Summary

Phase 11 removed the PySpark-specific `is_pyspark` branch from `validate()`, deleted `_handle_pyspark_validation_result`, removed a dead `nw.Implementation.PYSPARK` branch from `_materialize()`, added a `validate_collecting_errors()` helper in conftest.py, and updated all PySpark tests to use it. Documentation and capitalization nits were also applied.

The SE-01 architectural change is structurally sound — the unified `SchemaErrors` path works correctly for narwhals PySpark. However, two critical correctness defects were found:

1. `validate_collecting_errors()` swallows exceptions that originate from the **narwhals** backend on the **success** path. When the narwhals backend returns a valid PySpark DataFrame (no errors), the helper then calls `out_df.pandera.errors` on it. Under the narwhals backend the `.pandera` accessor is not populated, so this call raises `AttributeError` (or returns an accessor stub that diverges from the native backend). This means the success path is only reliable under the native backend; under narwhals all successful validations either return `{}` only by coincidence or raise silently-swallowed exceptions.

2. The `_cmp_errors` helper asserts `all(e["error"] for e in actual[key])` — this is a truthiness check, not a presence check — meaning an empty-string `error` field silently passes.

Four warnings and two info items are also documented below.

## Critical Issues

### CR-01: `validate_collecting_errors` success path crashes under narwhals backend

**File:** `tests/pyspark/conftest.py:218`
**Issue:** The success branch calls `out_df.pandera.errors` unconditionally. Under the narwhals backend, `schema.validate(df)` returns a native `pyspark.sql.DataFrame` **without** setting `.pandera.schema` or populating `.pandera.errors` (this is explicitly documented — "The .pandera.schema accessor is also not set by narwhals backend"). Accessing `out_df.pandera.errors` on a narwhals-returned DataFrame will therefore raise `AttributeError` (or return `None` if the accessor stub returns None). The `except SchemaErrors` handler does not catch `AttributeError`, so the exception leaks out of the helper, and every successful validation test that runs under the narwhals backend will fail with an unrelated `AttributeError`. The described behavior — "on success the dict is empty `{}`" — is only reliable when `out_df.pandera.errors` is `None` (native backend silent success), not when the accessor is missing entirely.

**Fix:** Branch on the active backend to choose whether to access `.pandera.errors`, or wrap the accessor access in a `try/except AttributeError`:

```python
def validate_collecting_errors(schema, df, **validate_kwargs):
    try:
        out_df = schema.validate(df, **validate_kwargs)
    except SchemaErrors as exc:
        # Narwhals path: rebuild the same nested dict structure from the exception.
        handler = ErrorHandler(lazy=True)
        handler.collect_errors(exc.schema_errors)
        schema_name = getattr(schema, "name", None) or getattr(
            schema, "__name__", None
        )
        errors = handler.summarize(schema_name=schema_name)
        return (None, dict(errors))

    # Success path: native backend attaches errors via accessor;
    # narwhals backend does not set .pandera.errors — treat as empty.
    try:
        errors = out_df.pandera.errors
    except AttributeError:
        errors = None
    return (out_df, dict(errors) if errors is not None else {})
```

Alternatively, check `CONFIG.use_narwhals_backend` explicitly and skip the accessor call.

---

### CR-02: `_cmp_errors` truth check does not distinguish missing `error` key from empty string

**File:** `tests/pyspark/conftest.py:247`
**Issue:** `assert all(e["error"] for e in actual[key])` evaluates the truthiness of the `"error"` value. An empty string `""` is falsy and would cause this assertion to fail silently (the `assert` would be `assert False` — which would raise, but the failure message would be opaque). More critically, if the `"error"` key is missing entirely, `e["error"]` raises `KeyError`, not a clean test failure. The assertion is also structurally backwards: it verifies that every error entry has a non-empty `error` field, but if the intent is to ignore the exact text while ensuring the field is populated, the correct check is `assert all("error" in e and e["error"] is not None for e in actual[key])`. The current form also does not catch cases where `error` is the integer `0` or another falsy non-empty value.

**Fix:**
```python
def _cmp_errors(actual, expected):
    def drop_error(entries):
        return [{k: v for k, v in e.items() if k != "error"} for e in entries]

    assert set(actual) == set(expected)
    for key in expected:
        assert drop_error(actual[key]) == drop_error(expected[key])
        # Verify 'error' field is present and non-None for all entries
        assert all("error" in e and e["error"] is not None for e in actual[key])
```

## Warnings

### WR-01: `validate_collecting_errors` docstring promises `{}` on narwhals success but the code cannot reach that return under narwhals

**File:** `tests/pyspark/conftest.py:196-231`
**Issue:** The docstring says "On success the dict is empty (`{}`)". But on the narwhals backend, the `try` block does `out_df.pandera.errors` on line 218 — which is the bug identified in CR-01. Once CR-01 is fixed, the docstring will be accurate. Until then the docstring misrepresents the actual behavior. This is also a maintenance risk: future readers may trust the docstring and not realize the accessor access is backend-conditional.

**Fix:** After fixing CR-01, update the docstring to explicitly note: "Under the narwhals backend, success returns `(out_df, {})` directly without accessor access."

---

### WR-02: `_to_frame_kind_nw` calls `.collect()` on native PySpark DataFrame's lazy result, but PySpark `nw.LazyFrame.collect()` triggers a Spark job

**File:** `pandera/backends/narwhals/container.py:78`
**Issue:** The `_to_frame_kind_nw` function checks `caller_was_eager_polars` to decide whether to call `native.collect()`. For a PySpark `nw.LazyFrame`, `nw.to_native(lf)` returns a native `pyspark.sql.DataFrame` (which PySpark itself treats as lazy). The `return_type` for a PySpark DataFrame is `pyspark.sql.DataFrame`, which:
- does NOT have a `collect` attribute at the class level (so `hasattr(return_type, "collect")` is `False`), AND
- its `__module__` is `pyspark.sql.dataframe` which starts with `pyspark`, not `polars`.

So `caller_was_eager_polars` evaluates to `False` for PySpark inputs (correct), and the function returns `native` (the native PySpark DataFrame). This works correctly today — but the variable name `caller_was_eager_polars` is actively misleading because the condition `not hasattr(return_type, "collect") and return_type.__module__.startswith("polars")` is the ONLY guard that prevents PySpark DataFrames from being misidentified as "eager polars". If a future framework's class has no `collect` at the class level and also has a module starting with `polars`, it would be misclassified. The logic should be made explicit.

**Fix:** Rename the variable or add a comment clarifying the two-condition guard:
```python
# Both conditions required:
# 1. No class-level .collect (distinguishes pl.DataFrame from pl.LazyFrame)
# 2. polars module prefix (distinguishes polars from PySpark — PySpark's module
#    starts with 'pyspark', not 'polars')
caller_was_eager_polars = (
    not hasattr(return_type, "collect")
    and return_type.__module__.startswith("polars")
)
```

---

### WR-03: `_concat_failure_cases` calls `nw.concat(nw_items)` on a mixed list that may contain `nw.DataFrame` and `nw.LazyFrame` items

**File:** `pandera/backends/narwhals/base.py:119`
**Issue:** In the Polars branch of `_concat_failure_cases`, `nw.concat(nw_items)` is called where `nw_items` is a list of items from `_build_lazy_failure_case`. For Polars inputs, `_build_lazy_failure_case` returns a `nw.LazyFrame`. But if the same code path is somehow reached with a mix of `nw.DataFrame` and `nw.LazyFrame` items in `nw_items` (e.g. if future code adds a path that produces `nw.DataFrame`), `nw.concat` would raise because narwhals does not allow mixing DataFrame and LazyFrame in a single `concat` call. There is no guard before the `nw.concat` call ensuring all items are the same type. This is currently safe because `_build_lazy_failure_case` always returns the same type for a given backend, but the absence of a guard is a latent correctness risk.

**Fix:** Add a type homogeneity assertion before the concat:
```python
# Polars lazy path
assert all(isinstance(i, nw.LazyFrame) for i in nw_items) or all(
    isinstance(i, nw.DataFrame) for i in nw_items
), "nw_items must be homogeneous (all LazyFrame or all DataFrame)"
lazy_result = nw.to_native(nw.concat(nw_items))
```

---

### WR-04: `test_pyspark_dataframeschema` uses `assert errors is not None` as success check, which is always true

**File:** `tests/pyspark/test_pyspark_container.py:45`
**Issue:** The test calls `validate_collecting_errors(schema, df)` with valid data and then asserts `assert errors is not None`. Since `validate_collecting_errors` always returns a `dict` (never `None`), this assertion is trivially true — even if validation is broken and returns wrong error content. The test provides no actual signal. The intent appears to be "errors should be empty" (`assert errors == {}`), matching the pattern used in `test_pyspark_dataframeschema` a few lines later (line 53: `assert errors2 == {}`).

**Fix:**
```python
_, errors = validate_collecting_errors(schema, df)
assert errors == {}  # valid data should produce no errors
```

## Info

### IN-01: Commented-out code block in `test_pyspark_check.py`

**File:** `tests/pyspark/test_pyspark_check.py:93-95`
**Issue:** Three lines of commented-out code remain:
```python
# if df_out.pandera.errors:
#     print(df_out.pandera.errors)
#     raise PysparkSchemaError
```
These were part of the old `.pandera.errors` pattern being replaced. They serve no documentation purpose and should be removed now that the migration to `validate_collecting_errors` is complete.

**Fix:** Delete lines 93-95 from `test_pyspark_check.py`.

---

### IN-02: `_materialize` docstring still references PySpark despite no PySpark branch

**File:** `pandera/api/narwhals/utils.py:44-62`
**Issue:** The docstring for `_materialize` contains: "Note: PySpark `nw.DataFrame` is always converted to `nw.LazyFrame` by `_to_lazy_nw` in `pandera/backends/narwhals/container.py` before `_materialize` is reached, so PySpark frames arrive as `nw.LazyFrame` and are handled by the first branch above." While architecturally accurate, this note is now the only PySpark-specific text in the file and serves as an implicit justification for why there is no PySpark branch. The DC-01 regression test (`test_materialize_has_no_pyspark_branch_after_dc01`) confirms the branch is gone. The docstring note is harmless but creates a subtle contract coupling between `_materialize` and the PySpark conversion happening in `container.py`. If `_to_lazy_nw` is ever changed, this note becomes misleading.

**Fix:** The note is acceptable as-is but could be shortened to: "PySpark frames are pre-converted to `nw.LazyFrame` by `_to_lazy_nw` before reaching this function."

---

_Reviewed: 2026-05-30_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
