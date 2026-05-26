# Phase 8: Test Quality Improvements - Pattern Map

**Mapped:** 2026-05-25
**Files analyzed:** 5 (4 modified, 1 production fix)
**Analogs found:** 5 / 5

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `tests/pyspark/conftest.py` | test-infrastructure | — | `tests/pyspark/conftest.py` (self — adding function) | self |
| `tests/pyspark/test_pyspark_error.py` | test | request-response | `tests/pyspark/test_pyspark_config.py` | exact |
| `tests/pyspark/test_pyspark_config.py` | test | request-response | `tests/pyspark/test_pyspark_config.py` (self — delegation refactor) | self |
| `tests/narwhals/test_arch03_schema_driven_dispatch.py` | test | request-response | `tests/narwhals/test_e2e.py` (PySpark fixture + guard pattern) | role-match |
| `pandera/backends/narwhals/base.py` | backend / utility | transform | `pandera/backends/narwhals/base.py` (self — PySpark branch is the model for polars branch fix) | self |

## Pattern Assignments

### `tests/pyspark/conftest.py` (adding module-level `_cmp_errors`)

**Analog:** `tests/pyspark/test_pyspark_config.py` (source of the function body)

**Existing conftest imports pattern** (lines 1-13):
```python
"""conftest"""

import datetime
import os

import pyspark
import pyspark.sql.types as T
import pytest
from packaging import version
from pyspark.sql import SparkSession

from pandera.config import PanderaConfig
```

**Function to add — verbatim from analog** (`test_pyspark_config.py` lines 34-47):
```python
def _cmp_errors(actual, expected):
    """Compare pandera error dicts ignoring the exact error message text.

    Error message format varies by backend (narwhals vs native PySpark),
    so only structural fields (check, column, schema) are compared.
    """

    def drop_error(entries):
        return [{k: v for k, v in e.items() if k != "error"} for e in entries]

    assert set(actual) == set(expected)
    for key in expected:
        assert drop_error(actual[key]) == drop_error(expected[key])
        assert all(e["error"] for e in actual[key])
```

**Placement:** Add after the existing `config_params` fixture at the bottom of `conftest.py` (line 192). No new imports needed — `_cmp_errors` has no imports.

**Naming rule (from feedback_naming_conventions):** The inner `drop_error` function must NOT use underscore prefix. The original `def drop_error(entries):` is already correct — preserve it verbatim.

---

### `tests/pyspark/test_pyspark_config.py` (delegation refactor of `TestPanderaConfig._cmp_errors`)

**Analog:** `tests/pyspark/test_pyspark_config.py` (self — lines 33-47)

**Current static method body** (lines 33-47):
```python
class TestPanderaConfig:
    @staticmethod
    def _cmp_errors(actual, expected):
        """Compare pandera error dicts ignoring the exact error message text.

        Error message format varies by backend (narwhals vs native PySpark),
        so only structural fields (check, column, schema) are compared.
        """

        def drop_error(entries):
            return [{k: v for k, v in e.items() if k != "error"} for e in entries]

        assert set(actual) == set(expected)
        for key in expected:
            assert drop_error(actual[key]) == drop_error(expected[key])
            assert all(e["error"] for e in actual[key])
```

**Target pattern — delegation body only:**
```python
@staticmethod
def _cmp_errors(actual, expected):
    """Delegates to module-level _cmp_errors in conftest."""
    _cmp_errors(actual, expected)
```

**Import note:** `test_pyspark_config.py` line 21 already has `from tests.pyspark.conftest import spark_df`. The module-level `_cmp_errors` from `conftest.py` is auto-available via conftest injection — no explicit import needed for the delegation call.

---

### `tests/pyspark/test_pyspark_error.py` (replace 6 CONFIG ternaries with `_cmp_errors`)

**Analog:** `tests/pyspark/test_pyspark_config.py` — `TestPanderaConfig._cmp_errors` usage pattern

**Current imports** (lines 1-16):
```python
"""Unit tests for dask_accessor module."""

import pyspark.sql.types as T
import pytest
from pyspark.sql.types import StringType

import pandera.pyspark as pa
from pandera.api.base import error_handler
from pandera.config import CONFIG
from pandera.errors import SchemaError, SchemaErrorReason
from pandera.pyspark import Column, DataFrameModel, DataFrameSchema, Field
from tests.pyspark.conftest import spark_df
```

**After refactor:** Remove `from pandera.config import CONFIG` if no other uses remain after replacing all 6 ternaries (verify with grep before removing).

**Site 1 — `test_pyspark_check_eq` DATA assertion (lines 55-79). BEFORE:**
```python
expected = {
    "DATAFRAME_CHECK": [
        {
            "check": "str_startswith('B')",
            "column": "product",
            "error": (
                "Check '<Check str_startswith: str_startswith('B')>' failed."
                if CONFIG.use_narwhals_backend
                else "<Schema Column(name=product, type=DataType(StringType()))> failed validation str_startswith('B')"
            ),
            "schema": "product_schema",
        },
        {
            "check": "greater_than(5)",
            "column": "price",
            "error": (
                "Check '<Check greater_than: greater_than(5)>' failed."
                if CONFIG.use_narwhals_backend
                else "<Schema Column(name=price, type=DataType(IntegerType()))> failed validation greater_than(5)"
            ),
            "schema": "product_schema",
        },
    ]
}
assert dict(df_out.pandera.errors["DATA"]) == expected
```

**AFTER:**
```python
expected = {
    "DATAFRAME_CHECK": [
        {
            "check": "str_startswith('B')",
            "column": "product",
            "schema": "product_schema",
        },
        {
            "check": "greater_than(5)",
            "column": "price",
            "schema": "product_schema",
        },
    ]
}
_cmp_errors(dict(df_out.pandera.errors["DATA"]), expected)
```

**Sites 2-3 — `test_pyspark_schema_data_checks` DATA assertion (lines 144-186). BEFORE:**
```python
expected = {
    "DATA": {
        "DATAFRAME_CHECK": [
            {
                "check": "str_startswith('B')",
                "column": "product",
                "error": (
                    "Check '<Check str_startswith: str_startswith('B')>' failed."
                    if CONFIG.use_narwhals_backend
                    else "..."
                ),
                "schema": "product_schema",
            },
            {
                "check": "greater_than(5)",
                "column": "price",
                "error": (
                    "Check '<Check greater_than: greater_than(5)>' failed."
                    if CONFIG.use_narwhals_backend
                    else "..."
                ),
                "schema": "product_schema",
            },
        ]
    },
    "SCHEMA": { ... }  # unchanged — no ternaries
}

assert dict(output_data.pandera.errors["DATA"]) == expected["DATA"]
assert dict(output_data.pandera.errors["SCHEMA"]) == expected["SCHEMA"]
```

**AFTER — DATA assertion only changes:**
```python
# expected["DATA"]["DATAFRAME_CHECK"] entries lose "error" key
# assertion becomes:
_cmp_errors(dict(output_data.pandera.errors["DATA"]), expected["DATA"])
assert dict(output_data.pandera.errors["SCHEMA"]) == expected["SCHEMA"]  # SCHEMA unchanged
```

**Sites 4-6 — `test_pyspark_fields` DATA assertion (lines 226-269) — same pattern as sites 2-3.**

**Key constraint:** SCHEMA assertions in all three tests (`WRONG_DATATYPE`, `SERIES_CONTAINS_NULLS`, etc.) use fixed error strings with no CONFIG ternaries. They keep `assert dict(...) == expected["SCHEMA"]` as-is.

---

### `tests/narwhals/test_arch03_schema_driven_dispatch.py` (delete 4 tests, add 2 PySpark-gated behavioral tests)

**Analog 1 (behavioral test structure):** `tests/narwhals/test_arch03_schema_driven_dispatch.py` lines 102-136 — existing 5th test (keep this one)

**Analog 2 (PySpark guard pattern):** `tests/narwhals/test_e2e.py` lines 58-69 + lines 706-742

**Tests to delete** (lines 26-99 — all 4 source-inspection tests):
- `test_check_dtype_has_no_is_pyspark_variable` (lines 26-44)
- `test_check_dtype_uses_pyspark_dtype_variable` (lines 47-59)
- `test_check_dtype_has_no_frame_implementation_probe_for_pyspark` (lines 62-80)
- `test_check_dtype_uses_pyspark_engine_isinstance_probe` (lines 83-99)

**Test to keep** (lines 102-136):
```python
def test_check_dtype_narwhals_schema_takes_narwhals_engine_path():
    """check_dtype with a narwhals-native dtype does NOT attempt PySpark operations."""
    from types import SimpleNamespace
    import narwhals.stable.v1 as nw
    import polars as pl
    from pandera.backends.narwhals.components import ColumnBackend
    from pandera.engines import narwhals_engine

    frame = nw.from_native(
        pl.LazyFrame({"col": [1, 2, 3]}), eager_or_interchange_only=False
    )
    schema = SimpleNamespace(
        selector="col", name="col", nullable=True, unique=False,
        dtype=narwhals_engine.Int64(), checks=[],
    )
    backend = ColumnBackend()
    results = backend.check_dtype(frame, schema)
    assert len(results) == 1
    assert results[0].passed is True
```

**PySpark guard pattern** (from `test_e2e.py` lines 58-69, 706-742):
```python
try:
    import pyspark.sql
    from pyspark.sql import SparkSession
    HAS_PYSPARK = True
except ImportError:
    HAS_PYSPARK = False

pyspark_only = pytest.mark.skipif(not HAS_PYSPARK, reason="pyspark not installed")
```

**`spark` fixture pattern** (from `test_e2e.py` lines 726-742 — copy inline into `test_arch03_schema_driven_dispatch.py`; the narwhals `conftest.py` does NOT define a `spark` fixture):
```python
@pytest.fixture(scope="module")
def spark():
    """Create a SparkSession for the module."""
    pytest.importorskip("pyspark")
    import pyspark
    from packaging import version

    PYSPARK_VERSION = version.parse(pyspark.__version__)
    builder = SparkSession.builder.config("spark.sql.ansi.enabled", False)
    if PYSPARK_VERSION >= version.parse("4.0.0"):
        builder = builder.config("spark.hadoop.fs.defaultFS", "file:///")
        builder = builder.config(
            "spark.sql.warehouse.dir", "file:///tmp/spark-warehouse"
        )
    spark_session = builder.getOrCreate()
    yield spark_session
    spark_session.stop()
```

**`_spark_env_vars` fixture pattern** (from `test_e2e.py` lines 706-723 — also needed for PySpark tests):
```python
@pytest.fixture(autouse=True, scope="function")
def _spark_env_vars():
    """Set environment variables required by PySpark before each test."""
    if not HAS_PYSPARK:
        yield
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

**New behavioral tests (NOT xfail — implementation is complete):**
```python
@pyspark_only
def test_check_dtype_pyspark_schema_pass(spark):
    """check_dtype with matching PySpark dtype passes (schema-driven dispatch)."""
    import pyspark.sql.types as T
    from types import SimpleNamespace
    import narwhals.stable.v1 as nw
    from pandera.backends.narwhals.components import ColumnBackend
    from pandera.engines import pyspark_engine

    df = spark.createDataFrame([(1,), (2,)], schema=["col"])
    frame = nw.from_native(df, eager_or_interchange_only=False)
    schema = SimpleNamespace(
        selector="col", name="col", nullable=True, unique=False,
        dtype=pyspark_engine.Engine.dtype(T.LongType()),
        checks=[],
    )
    backend = ColumnBackend()
    results = backend.check_dtype(frame, schema)
    assert len(results) == 1
    assert results[0].passed is True


@pyspark_only
def test_check_dtype_pyspark_schema_fail(spark):
    """check_dtype with mismatched PySpark dtype fails (schema-driven dispatch)."""
    import pyspark.sql.types as T
    from types import SimpleNamespace
    import narwhals.stable.v1 as nw
    from pandera.backends.narwhals.components import ColumnBackend
    from pandera.engines import pyspark_engine
    from pandera.errors import SchemaErrorReason

    df = spark.createDataFrame([(1,), (2,)], schema=["col"])  # LongType column
    frame = nw.from_native(df, eager_or_interchange_only=False)
    schema = SimpleNamespace(
        selector="col", name="col", nullable=True, unique=False,
        dtype=pyspark_engine.Engine.dtype(T.StringType()),  # wrong type
        checks=[],
    )
    backend = ColumnBackend()
    results = backend.check_dtype(frame, schema)
    assert len(results) == 1
    assert results[0].passed is False
    assert results[0].reason_code == SchemaErrorReason.WRONG_DATATYPE
```

**Pitfall:** Do NOT apply `@_xfail` from `test_components.py`. The `_xfail` marker there is conditioned on `ColumnBackend is None` (unimplemented). `ColumnBackend` IS implemented; new tests must pass immediately.

---

### `pandera/backends/narwhals/base.py` (fix `_concat_failure_cases` polars branch)

**Analog:** Lines 77-106 — the PySpark branch is the model for the pl_items-aware fix

**Current polars branch** (lines 107-109):
```python
elif first_nw.implementation == nw.Implementation.POLARS:
    # Polars lazy path: use nw.concat to stay lazy, then unwrap.
    return nw.to_native(nw.concat(nw_items))
```

**PySpark branch as structural model** (lines 77-106):
```python
if first_nw.implementation in (
    nw.Implementation.PYSPARK,
    nw.Implementation.PYSPARK_CONNECT,
):
    if pl_items:
        # ... warn and drop pl_items (SparkSession barrier)
    native_items = [nw.to_native(item) for item in nw_items]
    return functools.reduce(lambda a, b: a.union(b), native_items)
```

**Target fix for polars branch** (replace lines 107-109):
```python
elif first_nw.implementation == nw.Implementation.POLARS:
    # Polars lazy path: use nw.concat to stay lazy, then unwrap.
    # Collect and merge any native pl.DataFrame items (from schema-level
    # failures via _build_eager_failure_case / _build_scalar_failure_case)
    # that may coexist with lazy data-check failure frames.
    lazy_result = nw.to_native(nw.concat(nw_items))
    if pl_items:
        return pl.concat([lazy_result.collect()] + pl_items)
    return lazy_result
```

**Critical:** `nw.to_native(nw.concat(nw_items))` returns `pl.LazyFrame`. Call `.collect()` before passing to `pl.concat` — mixing `pl.LazyFrame` with `pl.DataFrame` in `pl.concat` is an error. No `SchemaWarning` — polars can merge both cleanly unlike the PySpark branch.

---

## Shared Patterns

### PySpark optional-dependency guard
**Source:** `tests/narwhals/test_e2e.py` lines 58-69
**Apply to:** `test_arch03_schema_driven_dispatch.py` (new addition)
```python
try:
    import pyspark.sql
    from pyspark.sql import SparkSession
    HAS_PYSPARK = True
except ImportError:
    HAS_PYSPARK = False

pyspark_only = pytest.mark.skipif(not HAS_PYSPARK, reason="pyspark not installed")
```

### `spark` fixture (module-scoped, narwhals tests)
**Source:** `tests/narwhals/test_e2e.py` lines 726-742
**Apply to:** `test_arch03_schema_driven_dispatch.py` (copy inline; narwhals `conftest.py` does not define one)
```python
@pytest.fixture(scope="module")
def spark():
    pytest.importorskip("pyspark")
    import pyspark
    from packaging import version
    PYSPARK_VERSION = version.parse(pyspark.__version__)
    builder = SparkSession.builder.config("spark.sql.ansi.enabled", False)
    if PYSPARK_VERSION >= version.parse("4.0.0"):
        builder = builder.config("spark.hadoop.fs.defaultFS", "file:///")
        builder = builder.config("spark.sql.warehouse.dir", "file:///tmp/spark-warehouse")
    spark_session = builder.getOrCreate()
    yield spark_session
    spark_session.stop()
```

### `_cmp_errors` helper (conftest-level, pyspark tests)
**Source:** `tests/pyspark/test_pyspark_config.py` lines 34-47
**Apply to:** `tests/pyspark/conftest.py` (destination), `tests/pyspark/test_pyspark_error.py` (consumer — no explicit import needed), `tests/pyspark/test_pyspark_config.py` (delegation call)

### DATA-only vs SCHEMA assertion split
**Source:** `tests/pyspark/test_pyspark_error.py` lines 183-186
**Apply to:** All three test functions modified in TQ-01
```python
# DATA: use _cmp_errors (structural comparison, drops error text)
_cmp_errors(dict(output_data.pandera.errors["DATA"]), expected["DATA"])
# SCHEMA: keep direct equality (error strings are backend-invariant)
assert dict(output_data.pandera.errors["SCHEMA"]) == expected["SCHEMA"]
```

### `SimpleNamespace` schema stub for unit tests
**Source:** `tests/narwhals/test_arch03_schema_driven_dispatch.py` lines 109-127
**Apply to:** New PySpark-gated behavioral tests in same file
```python
from types import SimpleNamespace
schema = SimpleNamespace(
    selector="col",
    name="col",
    nullable=True,
    unique=False,
    dtype=<engine>.dtype(<type>()),
    checks=[],
)
```

## No Analog Found

No files in this phase are without analogs. All five targets have clear matches in the existing codebase.

## Metadata

**Analog search scope:** `tests/pyspark/`, `tests/narwhals/`, `pandera/backends/narwhals/`
**Files scanned:** 7 (test_pyspark_config.py, test_pyspark_error.py, conftest.py × 2, test_arch03_schema_driven_dispatch.py, test_e2e.py, base.py)
**Pattern extraction date:** 2026-05-25
