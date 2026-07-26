"""Tests for the pyarrow DataFrameSchema API."""

import pyarrow
import pyarrow.compute as pc
import pytest

import pandera.pyarrow as pa
from pandera.config import ValidationDepth, config_context
from pandera.errors import SchemaError, SchemaErrors, SchemaWarning


@pytest.fixture
def table():
    return pyarrow.table(
        {
            "int_col": [1, 2, 3],
            "float_col": [1.0, 2.0, 3.0],
            "str_col": ["a", "b", "c"],
        }
    )


@pytest.fixture
def schema():
    return pa.DataFrameSchema(
        {
            "int_col": pa.Column(int, pa.Check.gt(0)),
            "float_col": pa.Column(float),
            "str_col": pa.Column(str),
        }
    )


def test_validate_returns_pyarrow_table(table, schema):
    validated = schema.validate(table)
    assert isinstance(validated, pyarrow.Table)
    assert validated.equals(table)


def test_data_level_check_failure_is_raised(schema):
    """Regression test: data-level checks must not be silently skipped.

    A pyarrow.Table is always materialized, so the default validation depth
    is SCHEMA_AND_DATA — the same as pl.DataFrame, not pl.LazyFrame.
    """
    invalid = pyarrow.table(
        {
            "int_col": [1, -2, 3],
            "float_col": [1.0, 2.0, 3.0],
            "str_col": ["a", "b", "c"],
        }
    )
    with pytest.raises(SchemaError, match="greater_than"):
        schema.validate(invalid)


def test_default_validation_depth_is_schema_and_data(table):
    from pandera.api.pyarrow.utils import get_validation_depth

    assert get_validation_depth(table) is ValidationDepth.SCHEMA_AND_DATA


def test_validation_depth_config_is_respected(schema):
    invalid = pyarrow.table(
        {
            "int_col": [1, -2, 3],
            "float_col": [1.0, 2.0, 3.0],
            "str_col": ["a", "b", "c"],
        }
    )
    with config_context(validation_depth=ValidationDepth.SCHEMA_ONLY):
        assert schema.validate(invalid).equals(invalid)


def test_wrong_dtype_raises(schema):
    invalid = pyarrow.table(
        {
            "int_col": [1.5, 2.5, 3.5],
            "float_col": [1.0, 2.0, 3.0],
            "str_col": ["a", "b", "c"],
        }
    )
    with pytest.raises(SchemaError):
        schema.validate(invalid)


def test_missing_column_raises(schema):
    with pytest.raises(SchemaError):
        schema.validate(pyarrow.table({"int_col": [1, 2, 3]}))


def test_strict_rejects_extra_column(table):
    schema = pa.DataFrameSchema({"int_col": pa.Column(int)}, strict=True)
    with pytest.raises(SchemaError):
        schema.validate(table)


def test_strict_filter_drops_extra_columns(table):
    schema = pa.DataFrameSchema({"int_col": pa.Column(int)}, strict="filter")
    validated = schema.validate(table)
    assert validated.column_names == ["int_col"]


def test_lazy_collects_all_errors():
    schema = pa.DataFrameSchema(
        {
            "a": pa.Column(int, pa.Check.gt(10)),
            "b": pa.Column(str),
        }
    )
    invalid = pyarrow.table({"a": [1, 2], "b": [1, 2]})
    with pytest.raises(SchemaErrors) as exc_info:
        schema.validate(invalid, lazy=True)
    assert len(exc_info.value.failure_cases) >= 2


def test_failure_cases_are_a_pyarrow_table():
    """failure_cases must come back as pyarrow, not depend on polars.

    The narwhals eager failure-case builder round-trips through polars. A
    pyarrow install has no polars, so pyarrow gets its own builder — and the
    reported type must not change based on whether polars happens to be
    installed alongside.
    """
    schema = pa.DataFrameSchema({"a": pa.Column(int, pa.Check.gt(0))})
    with pytest.raises(SchemaErrors) as exc_info:
        schema.validate(pyarrow.table({"a": [1, -2, -3]}), lazy=True)

    failure_cases = exc_info.value.failure_cases
    assert isinstance(failure_cases, pyarrow.Table)
    assert set(failure_cases.column_names) >= {
        "failure_case",
        "schema_context",
        "column",
        "check",
        "check_number",
        "index",
    }
    assert failure_cases.column("failure_case").to_pylist() == ["-2", "-3"]
    assert failure_cases.column("column").to_pylist() == ["a", "a"]
    # index is null on the deferred-expr path, matching the polars backend
    assert failure_cases.column("index").to_pylist() == [None, None]


def test_failure_cases_include_schema_level_errors():
    """Scalar (schema-level) and row-level failure cases must concatenate.

    They are built by different code paths; if they land on different
    backends the concat step blows up.
    """
    schema = pa.DataFrameSchema(
        {
            "a": pa.Column(int, pa.Check.gt(0)),
            "missing": pa.Column(str),
        }
    )
    with pytest.raises(SchemaErrors) as exc_info:
        schema.validate(pyarrow.table({"a": [1, -2]}), lazy=True)

    failure_cases = exc_info.value.failure_cases
    assert isinstance(failure_cases, pyarrow.Table)
    assert failure_cases.num_rows == 2
    assert set(failure_cases.column("failure_case").to_pylist()) == {
        "missing",
        "-2",
    }
    assert set(failure_cases.column("schema_context").to_pylist()) == {
        "DataFrameSchema",
        "Column",
    }


def test_nullable_false_rejects_nulls():
    schema = pa.DataFrameSchema({"a": pa.Column(int, nullable=False)})
    with pytest.raises(SchemaError):
        schema.validate(pyarrow.table({"a": [1, None, 3]}))


def test_nullable_true_allows_nulls():
    schema = pa.DataFrameSchema({"a": pa.Column(int, nullable=True)})
    tbl = pyarrow.table({"a": [1, None, 3]})
    assert schema.validate(tbl).equals(tbl)


def test_unique_constraint():
    schema = pa.DataFrameSchema({"a": pa.Column(int, unique=True)})
    assert schema.validate(pyarrow.table({"a": [1, 2, 3]})) is not None
    with pytest.raises(SchemaError):
        schema.validate(pyarrow.table({"a": [1, 1, 3]}))


def test_column_coerce_is_not_supported():
    """Column-level coerce is a documented gap in the narwhals backend.

    pyarrow is served exclusively by the narwhals backends, where coercion is
    deferred to v2 (see the strict xfail markers in
    ``tests/narwhals/test_parity.py``). Assert the warned no-op behaviour so
    this test flips loudly when coerce lands.
    """
    schema = pa.DataFrameSchema({"a": pa.Column(str, coerce=True)})
    with pytest.warns(SchemaWarning, match="coerce=True is not applied"):
        with pytest.raises(SchemaError):
            schema.validate(pyarrow.table({"a": [1, 2, 3]}))


def test_regex_column():
    schema = pa.DataFrameSchema({"^val_.+$": pa.Column(int, regex=True)})
    tbl = pyarrow.table({"val_a": [1], "val_b": [2], "other": ["x"]})
    assert schema.validate(tbl).equals(tbl)

    bad = pyarrow.table({"val_a": [1], "val_b": ["x"]})
    with pytest.raises(SchemaError):
        schema.validate(bad)


def test_drop_invalid_rows():
    schema = pa.DataFrameSchema(
        {"a": pa.Column(int, pa.Check.gt(0))}, drop_invalid_rows=True
    )
    validated = schema.validate(pyarrow.table({"a": [1, -2, 3]}), lazy=True)
    assert validated.column("a").to_pylist() == [1, 3]


def test_dataframe_level_check():
    schema = pa.DataFrameSchema(
        {"a": pa.Column(int), "b": pa.Column(int)},
        checks=pa.Check(
            lambda data: pc.less(data.table["a"], data.table["b"]),
        ),
    )
    assert schema.validate(pyarrow.table({"a": [1], "b": [2]})) is not None
    with pytest.raises(SchemaError):
        schema.validate(pyarrow.table({"a": [3], "b": [2]}))


def test_validation_disabled(table, schema):
    invalid = pyarrow.table(
        {
            "int_col": [-1],
            "float_col": [1.0],
            "str_col": ["a"],
        }
    )
    with config_context(validation_enabled=False):
        assert schema.validate(invalid) is invalid


def test_data_synthesis_not_supported(schema):
    with pytest.raises(NotImplementedError):
        schema.example()
    with pytest.raises(NotImplementedError):
        schema.strategy()
