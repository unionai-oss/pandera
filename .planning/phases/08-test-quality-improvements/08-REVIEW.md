---
phase: 08-test-quality-improvements
reviewed: 2026-05-26T00:00:00Z
depth: standard
files_reviewed: 7
files_reviewed_list:
  - tests/pyspark/conftest.py
  - tests/pyspark/test_pyspark_config.py
  - tests/pyspark/test_pyspark_error.py
  - tests/narwhals/test_concat_failure_cases.py
  - pandera/backends/narwhals/base.py
  - tests/narwhals/test_arch03_schema_driven_dispatch.py
  - pandera/backends/pyspark/register.py
findings:
  critical: 3
  warning: 5
  info: 2
  total: 10
status: issues_found
---

# Phase 08: Code Review Report

**Reviewed:** 2026-05-26T00:00:00Z
**Depth:** standard
**Files Reviewed:** 7
**Status:** issues_found

## Summary

The phase 08 work adds regression tests for `_concat_failure_cases`, behavioral tests for ARCH-03 schema-driven dispatch, and cleans up PySpark test infrastructure (`conftest.py`, `test_pyspark_config.py`, `test_pyspark_error.py`). The narwhals `base.py` carries a bug in `run_check` that predates this phase but remains unaddressed. Three findings are blockers: a wrong-value assignment in `run_check` that silently corrupts `failure_cases` data, a silent data-loss path in `_concat_failure_cases` for SQL-lazy backends, and a test brittleness pattern that breaks when switching between native-PySpark and narwhals backends. Five warnings cover test design issues and dead code.

---

## Critical Issues

### CR-01: `failure_cases = passed` assigns a `bool` instead of `None` — corrupts failure case data

**File:** `pandera/backends/narwhals/base.py:233`
**Issue:** In `run_check`, the `else` branch (reached when `check_result.failure_cases is None` and `check_result.check_output` is _not_ a `nw.Expr`) assigns `failure_cases = passed`. `passed` is a Python `bool` (`False` at this point), so `failure_cases` becomes `False`. This value flows into `failure_cases_metadata` as `err.failure_cases = False`. There, `nw.from_native(False, ...)` raises `TypeError`; the `except TypeError: pass` silently catches it; then `_build_scalar_failure_case` is called with `err.failure_cases = False`, recording the literal value `False` as the failure case instead of the real failing data.

This path is reached when a check function returns a plain `bool` (not a `nw.Expr` or `nw.LazyFrame`): `postprocess_bool_output` returns a `CheckResult` with `failure_cases=None` and `check_output=nw.LazyFrame`. In `run_check`, that `check_output` is not a `nw.Expr`, so the `else` branch fires.

**Fix:**
```python
# Replace line 233:
else:
    failure_cases = None   # No structured failure cases available for scalar-bool checks
```

---

### CR-02: SQL-lazy `else` branch in `_concat_failure_cases` silently drops `pl_items` without warning

**File:** `pandera/backends/narwhals/base.py:123-126`
**Issue:** The PySpark branch (lines 87–108) emits a `SchemaWarning` and explains that `pl_items` (scalar polars frames from `_build_scalar_failure_case`) are dropped because they cannot be converted to PySpark without a `SparkSession`. The Polars branch (lines 120-122) correctly merges `pl_items` with `pl.concat`. However, the SQL-lazy `else` branch for ibis/DuckDB/SQLFrame (lines 124-126) processes only `nw_items` via `nw.to_native` + `union`, and ignores `pl_items` entirely — no warning, no merge, silent loss. Schema-level failure cases (from `_build_scalar_failure_case`) are silently omitted from the combined `failure_cases` frame for ibis/DuckDB backends.

**Fix:**
```python
else:
    # SQL-lazy path (ibis, DuckDB, etc.): unwrap to native and union.
    if pl_items:
        # pl.DataFrame scalar items cannot be converted to a SQL-lazy table
        # without a backend connection — warn and drop, same as PySpark.
        dropped_info = []
        for item in pl_items:
            if isinstance(item, pl.DataFrame) and "column" in item.columns:
                dropped_info.extend(item["column"].to_list())
        warnings.warn(
            "Some schema-level failure cases (columns: "
            + repr(dropped_info)
            + ") could not be included in the SQL-lazy failure_cases output "
            "because scalar polars frames cannot be converted to a SQL-lazy "
            "table without a backend connection.",
            SchemaWarning,
            stacklevel=3,
        )
    native_items = [nw.to_native(item) for item in nw_items]
    return functools.reduce(lambda a, b: a.union(b), native_items)
```

---

### CR-03: Direct `dict` equality for SCHEMA error assertions includes exact `"error"` message text — breaks when narwhals backend is active

**File:** `tests/pyspark/test_pyspark_error.py:102, 165, 238`
**Issue:** Three tests assert SCHEMA error dicts with exact `"error"` message strings using `==`:

- `test_pyspark_check_nullable` (line 102): expected `"error": "non-nullable column 'price' contains null"`
- `test_pyspark_schema_data_checks` (line 165): expected contains exact `"error"` text for `WRONG_DATATYPE`
- `test_pyspark_fields` (line 238): same pattern for `WRONG_DATATYPE`

`_cmp_errors` was introduced specifically to avoid this fragility for DATA errors, but SCHEMA errors still use raw equality. If the narwhals backend produces a different error-message format from the native PySpark backend (e.g. different capitalization, type rendering, or punctuation), all three tests will fail. The `pytestmark` parametrize runs these with both `spark` and `spark_connect`, but neither session exercises the narwhals backend path in its error-message formatter.

**Fix:** Use `_cmp_errors` for SCHEMA error comparisons just as DATA errors already do:
```python
# test_pyspark_check_nullable line 102:
_cmp_errors(
    dict(dataframe_output.pandera.errors["SCHEMA"]),
    {"SERIES_CONTAINS_NULLS": [{"check": "not_nullable", "column": "price", "schema": None}]},
)

# test_pyspark_schema_data_checks line 165:
_cmp_errors(
    dict(output_data.pandera.errors["SCHEMA"]),
    expected["SCHEMA"],  # after removing "error" keys from expected
)

# test_pyspark_fields line 238:
_cmp_errors(schema_errors, expected["SCHEMA"])  # after removing "error" keys
```

---

## Warnings

### WR-01: `test_pyspark__error_handler_lazy_validation` and `test_cache_dataframe_settings` accept `spark_session` but never use it — phantom parametrization

**File:** `tests/pyspark/test_pyspark_error.py:241-244`, `tests/pyspark/test_pyspark_config.py:334-355`
**Issue:** Both functions receive `spark_session` (a string — `"spark"` or `"spark_connect"`) via the module-level `pytestmark` parametrize, but neither calls `request.getfixturevalue(spark_session)` or uses the value in any way. The test bodies test pure-Python behavior (error handler state and config context, respectively). As a result each test runs twice identically, wasting CI time and obscuring the actual coverage intent.

**Fix:** Remove `spark_session` from the parameter list of both functions and apply a local override to suppress the module-level parametrize (or move these tests to a non-spark-parametrized class/function):
```python
@pytest.mark.parametrize("spark_session", [None])  # suppress module-level parametrize
def test_cache_dataframe_settings(self, cache_dataframe, keep_cached_dataframe):
    ...

@pytest.mark.parametrize("spark_session", [None])
def test_pyspark__error_handler_lazy_validation():
    ...
```

---

### WR-02: `Optional` imported but never used in `register.py`

**File:** `pandera/backends/pyspark/register.py:4`
**Issue:** `from typing import Optional` is present at the top of the file, but `Optional` is never referenced in the module. The type annotation on line 23 uses the PEP 604 union syntax `str | None`, making the import dead.

**Fix:**
```python
# Remove line 4:
# from typing import Optional
```

---

### WR-03: `check_cls_fqn` parameter in `register_pyspark_backends` is unused dead code

**File:** `pandera/backends/pyspark/register.py:23`
**Issue:** `check_cls_fqn: str | None = None` is declared as a parameter and participates in the `lru_cache` key, but its value is never read inside the function body. All three call sites (`container.py:41`, `column_schema.py:75`, `components.py:107`) invoke `register_pyspark_backends()` with no arguments. The parameter exists only as a potential future extension point but is currently misleading — it implies callers can vary behavior per `check_cls_fqn`, which they cannot.

**Fix:** Remove the parameter:
```python
@lru_cache
def register_pyspark_backends():
    ...
```

---

### WR-04: `conftest.spark_env_vars` sets environment variables without teardown — pollutes test isolation

**File:** `tests/pyspark/conftest.py:17-21`
**Issue:** The `spark_env_vars` autouse fixture sets `SPARK_LOCAL_IP` and `PYARROW_IGNORE_TIMEZONE` but does not restore or unset them after the test. It is a plain function, not a `yield` fixture. Any test that runs after the PySpark suite in the same pytest session will inherit these environment variables. Contrast with `tests/narwhals/test_arch03_schema_driven_dispatch.py` which correctly saves and restores env var values in its equivalent `_spark_env_vars` fixture.

**Fix:**
```python
@pytest.fixture(autouse=True)
def spark_env_vars():
    """Sets environment variables for pyspark."""
    prev = {
        "SPARK_LOCAL_IP": os.environ.get("SPARK_LOCAL_IP"),
        "PYARROW_IGNORE_TIMEZONE": os.environ.get("PYARROW_IGNORE_TIMEZONE"),
    }
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
    os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"
    yield
    for k, v in prev.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
```

---

### WR-05: `TestPanderaConfig._cmp_errors` is a redundant static method wrapper

**File:** `tests/pyspark/test_pyspark_config.py:33-36`
**Issue:** `TestPanderaConfig._cmp_errors` is a `@staticmethod` that does nothing except delegate to the module-level `_cmp_errors` from conftest. This adds a layer of indirection with no benefit: the method cannot override behavior, cannot be subclassed usefully, and callers inside `TestPanderaConfig` could call `_cmp_errors(actual, expected)` directly. The name collision between the static method and the imported conftest helper is also confusing.

**Fix:** Remove `TestPanderaConfig._cmp_errors` (lines 33–36) and replace all `self._cmp_errors(...)` calls with direct `_cmp_errors(...)` calls.

---

## Info

### IN-01: Commented-out code left in `test_schema_and_data`

**File:** `tests/pyspark/test_pyspark_config.py:235`
**Issue:** `# self.remove_python_module_cache()` is a commented-out call to a method that does not exist anywhere in the file. It's dead code from a prior refactor.

**Fix:** Remove line 235.

---

### IN-02: TODO comment referencing ARCH-02 follow-up left in production code

**File:** `pandera/backends/narwhals/base.py:100-103`
**Issue:** A multi-line `TODO(ARCH-02 follow-up)` comment inside a `SchemaWarning` string references a planned improvement (SparkSession-mediated conversion) without a tracking issue number or milestone. It is acceptable to leave TODOs during active development but this one is in a user-visible warning message string, which is unusual.

**Fix:** Move the TODO to a code comment above the `warnings.warn()` call and trim the warning message to user-relevant content only:
```python
# TODO(ARCH-02 follow-up): SparkSession-mediated conversion (Approach B) when
# a SparkSession reference is available. See project tracking for resolution.
warnings.warn(
    "Some schema-level failure cases (columns: "
    + repr(dropped_info)
    + ") could not be included in the PySpark failure_cases output because "
    "scalar polars frames cannot be converted to PySpark without a SparkSession. "
    "These schema errors are still reported in df.pandera.errors.",
    SchemaWarning,
    stacklevel=3,
)
```

---

_Reviewed: 2026-05-26T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
