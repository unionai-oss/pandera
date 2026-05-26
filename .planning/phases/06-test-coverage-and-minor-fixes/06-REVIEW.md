---
phase: 06-test-coverage-and-minor-fixes
reviewed: 2026-05-25T00:00:00Z
depth: standard
files_reviewed: 9
files_reviewed_list:
  - .github/workflows/ci-tests.yml
  - pandera/api/pyspark/types.py
  - pandera/backends/ibis/container.py
  - pandera/backends/narwhals/container.py
  - tests/ibis/test_ibis_container.py
  - tests/narwhals/conftest.py
  - tests/narwhals/test_e2e.py
  - tests/pyspark/test_pyspark_model.py
  - tests/pyspark/test_pyspark_narwhals_register.py
findings:
  critical: 2
  warning: 5
  info: 3
  total: 10
status: issues_found
---

# Phase 06: Code Review Report

**Reviewed:** 2026-05-25
**Depth:** standard
**Files Reviewed:** 9
**Status:** issues_found

## Summary

This phase added CI workflow updates, minor type fixes, ibis and narwhals backend container
implementations, and a suite of new tests (narwhals e2e, pyspark model tests, narwhals register
tests). The production backend code contains two critical bugs copied from an incorrect pattern
relative to the pandas reference backend. Five test quality defects (including a missing assertion
that silently ignores a filter-strictness test) and three info-level issues round out the findings.

---

## Critical Issues

### CR-01: `strict_filter_columns` — stale / unbound `next_ordered_col` after `StopIteration`

**File:** `pandera/backends/narwhals/container.py:541-554` and `pandera/backends/ibis/container.py:326-339`

**Issue:** Both backends use the pattern:

```python
try:
    next_ordered_col = next(sorted_column_names)
except StopIteration:
    pass            # ← falls through; next_ordered_col is NOT updated
if next_ordered_col != column:   # ← runs unconditionally
    raise SchemaError(...)
```

When `StopIteration` fires (the sorted iterator is exhausted), `pass` does nothing and execution
falls to the `if` comparison on the very next line. Two failure modes result:

1. **First-iteration exhaustion:** `next_ordered_col` has never been assigned →
   `UnboundLocalError` / `NameError` propagates as an unhandled exception, hiding the actual
   schema problem.
2. **Mid-loop exhaustion:** `next_ordered_col` retains its value from the previous iteration →
   the comparison fires against a stale column name. A correctly-ordered column receives a
   spurious `COLUMN_NOT_ORDERED` `SchemaError`.

The pandas reference backend (`pandera/backends/pandas/container.py:569-593`) implements this
correctly using `try / except / else` — the comparison only runs inside the `else` clause,
i.e. only when `StopIteration` was **not** raised.

**Fix:** Restructure both backends to mirror the pandas pattern:

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

Apply the same fix to `pandera/backends/ibis/container.py:326-339`.

---

### CR-02: `run_schema_component_checks` — `assert all(check_passed)` is an incorrect production guard

**File:** `pandera/backends/narwhals/container.py:367` and `pandera/backends/ibis/container.py:208`

**Issue:** Both backends use a bare `assert` statement to verify that all schema component
validations "passed":

```python
assert all(check_passed)
```

Two problems:

1. **`assert` is silently removed in optimised builds** (`python -O` / `PYTHONOPTIMIZE=1`), so
   the guard becomes a no-op.  In CPython distributions where optimisation is enabled by default
   (common in some Docker/Spark images), the check is skipped entirely.
2. **`all([])` is `True`.** When every component validation raises `SchemaError` or
   `SchemaErrors`, each exception is caught and its result is appended to `check_results`; nothing
   is appended to `check_passed`. The loop exits with `check_passed = []`, the assert passes, and
   `check_results` containing all failures is returned — which is the intended behaviour.
   However, if `schema_component.validate()` ever returns `None` legitimately (or a falsy value),
   `check_passed` contains `False`, the `assert` raises `AssertionError` (not `SchemaError`),
   and the error message exposed to the user is entirely misleading.

The assert was presumably copied from the pandas backend where returning `None` is also a signal
of success for some backends; the comment in the narwhals version acknowledges "The component
validate() not raising is the success signal" — making the boolean list redundant.

**Fix:** Remove `check_passed` entirely and delete the `assert`. The error collection loop already
handles all failure cases via the `except` branches:

```python
def run_schema_component_checks(self, check_obj, schema, schema_components, lazy):
    check_results = []
    native_obj = _to_native(check_obj)
    for schema_component in schema_components:
        try:
            schema_component.validate(native_obj, lazy=lazy)
        except SchemaError as err:
            check_results.append(
                CoreCheckResult(
                    passed=False,
                    check="schema_component_checks",
                    reason_code=err.reason_code,
                    schema_error=err,
                )
            )
        except SchemaErrors as err:
            check_results.extend(
                CoreCheckResult(
                    passed=False,
                    check="schema_component_checks",
                    reason_code=schema_error.reason_code,
                    schema_error=schema_error,
                )
                for schema_error in err.schema_errors
            )
    return check_results
```

Apply the same fix to `pandera/backends/ibis/container.py:172-209`.

---

## Warnings

### WR-01: Missing `assert` in `test_strict_filter` — the "filter" branch is never actually verified

**File:** `tests/ibis/test_ibis_container.py:151`

**Issue:** The assertion that `strict="filter"` drops extra columns is missing:

```python
# setting strict to "filter" should remove the extra column
t_schema_basic.strict = "filter"
filtered_data = modified_data.pipe(t_schema_basic.validate)
filtered_data.execute().equals(t_basic.execute())   # ← result discarded!
```

`filtered_data.execute().equals(t_basic.execute())` returns a `bool` that is never
asserted. The test silently passes whether or not the filter actually works.

**Fix:**
```python
assert filtered_data.execute().equals(t_basic.execute())
```

---

### WR-02: Dead code in `test_drop_invalid_rows` — `got` and `expected` computed but not used

**File:** `tests/ibis/test_ibis_container.py:444-445`

**Issue:**
```python
got = validated_data.execute()
expected = expected_valid_data.execute()
assert validated_data.execute().equals(expected_valid_data.execute())
```

`got` and `expected` are each computed by calling `.execute()` (triggering a SQL round-trip)
and then immediately discarded. The actual assertion re-computes both via `.execute()` again,
meaning the data is materialised three times instead of once. The variables are dead code.

**Fix:** Remove the dead assignments and use the already-computed variables in the assertion:
```python
got = validated_data.execute()
expected = expected_valid_data.execute()
assert got.equals(expected)
```

---

### WR-03: Duplicate parametrize value in `test_different_unique_settings`

**File:** `tests/ibis/test_ibis_container.py:220-223`

**Issue:** `("exclude_first", [4, 5, 6, 7])` appears **twice** in the parametrize list:

```python
@pytest.mark.parametrize(
    "unique,answers",
    [
        ("exclude_first", [4, 5, 6, 7]),   # first occurrence
        ("all", [0, 1, 2, 4, 5, 6, 7]),
        ("exclude_first", [4, 5, 6, 7]),   # DUPLICATE — adds no coverage
        ("exclude_last", [0, 1, 2, 4]),
    ],
)
```

The duplicate generates an identical test run, wasting CI time with no additional coverage.
Based on the comment ("default is to report all unique violations **except the first**"), the
first entry was likely intended to be `"exclude_last"` or some other setting, making the
duplicate a latent documentation error that may also indicate missing coverage of a case.

**Fix:** Determine which setting was intended for the first entry and replace the duplicate:
```python
@pytest.mark.parametrize(
    "unique,answers",
    [
        ("exclude_last", [0, 1, 2, 4]),      # verify this mapping is correct
        ("all", [0, 1, 2, 4, 5, 6, 7]),
        ("exclude_first", [4, 5, 6, 7]),
        ("exclude_last", [0, 1, 2, 4]),
    ],
)
```

---

### WR-04: `_spark_env_vars` autouse fixture sets environment variables without cleanup

**File:** `tests/narwhals/test_e2e.py:706-714`

**Issue:**
```python
@pytest.fixture(autouse=True, scope="function")
def _spark_env_vars():
    """Set environment variables required by PySpark before each test."""
    if HAS_PYSPARK:
        os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
        os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"
    # ← no yield, no cleanup
```

The fixture sets `SPARK_LOCAL_IP` and `PYARROW_IGNORE_TIMEZONE` in `os.environ` and returns
without restoring them. Once any test runs (and PySpark is installed), both variables leak
into all subsequent tests in the same process — including polars and ibis tests that have no
relation to PySpark or PyArrow timezone handling. `PYARROW_IGNORE_TIMEZONE=1` in particular
can mask timezone-related correctness issues in non-PySpark tests.

**Fix:** Store and restore the prior values:
```python
@pytest.fixture(autouse=True, scope="function")
def _spark_env_vars():
    if not HAS_PYSPARK:
        return
    prev = {k: os.environ.get(k) for k in ("SPARK_LOCAL_IP", "PYARROW_IGNORE_TIMEZONE")}
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

### WR-05: `TestCustomChecksIbis.test_schema_level_check_passes` — name and docstring contradict the test body

**File:** `tests/narwhals/test_e2e.py:487-499`

**Issue:** The test is named `test_schema_level_check_passes` and carries the docstring
`"Schema-level ibis check (key='*') passes on valid data."` However, the test body uses data
where `x < y` everywhere and expects `SchemaError` to be raised — i.e., the check **fails**,
not passes. The comment inside even acknowledges: `"ibis_table has x=[1,2,3], y=[10,20,30], so
x < y everywhere"`. This is a failure-path test masquerading as a pass-path test.

This creates a confusing test suite: a reader expecting "passes" tests to not use
`pytest.raises(SchemaError)` will waste time debugging.

**Fix:** Rename the test and update its docstring to reflect what it actually tests:
```python
def test_schema_level_check_fails_on_invalid_data(self, ibis_table):
    """Schema-level ibis check (key='*') raises SchemaError when check condition fails."""
    ...
```

---

## Info

### IN-01: Duplicate `type` entry in `PySparkDtypeInputTypes`

**File:** `pandera/api/pyspark/types.py:65-84`

**Issue:** `type` appears twice in the `PySparkDtypeInputTypes` `Union`:

```python
PySparkDtypeInputTypes = Union[
    str,
    int,
    float,
    bool,
    type,     # first occurrence (line 69)
    DataType,
    type,     # duplicate (line 71)
    ...
]
```

Python's `typing.Union` deduplicates at runtime, so this is functionally harmless, but it
indicates a copy-paste error and will confuse anyone reading the type definition.

**Fix:** Remove the second `type` entry from the Union.

---

### IN-02: `config_context()` with no arguments used as a no-op context manager

**File:** `pandera/backends/narwhals/container.py:175`

**Issue:**
```python
_check_ctx = (
    config_context(validation_depth=ValidationDepth.SCHEMA_AND_DATA)
    if getattr(schema, "drop_invalid_rows", False)
    else config_context()       # ← no-op: all args are None, no effect
)
```

`config_context()` with all-default arguments still enters/exits the context manager machinery
and calls `get_config_context()` + `reset_config_context()` on every validation call, even
when `drop_invalid_rows=False`. `contextlib.nullcontext()` is the idiomatic replacement for a
no-op context manager.

**Fix:**
```python
from contextlib import nullcontext

_check_ctx = (
    config_context(validation_depth=ValidationDepth.SCHEMA_AND_DATA)
    if getattr(schema, "drop_invalid_rows", False)
    else nullcontext()
)
```

---

### IN-03: `unit-tests-narwhals` CI job sets `shell: pwsh` as default on `ubuntu-latest`

**File:** `.github/workflows/ci-tests.yml:385`

**Issue:**
```yaml
unit-tests-narwhals:
  runs-on: ${{ matrix.os }}   # currently only ubuntu-latest
  defaults:
    run:
      shell: pwsh              # PowerShell on Linux
```

`pwsh` is available on GitHub-hosted Ubuntu runners, but it is unexpected and inconsistent:
all other Linux jobs in this file use `bash` or `bash -l {0}`. The `printenv | sort` step
uses `shell: bash` (explicit override on that step), confirming the default `pwsh` was likely
a copy-paste from a Windows job. PowerShell syntax differs from bash for environment variables
and quoting, which could cause subtle issues if new steps are added without explicit `shell:` overrides.

**Fix:** Change the default shell to `bash` for this job to match all other Ubuntu jobs:
```yaml
defaults:
  run:
    shell: bash
```

---

_Reviewed: 2026-05-25_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
