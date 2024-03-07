"""Unit tests for polars components."""

import pytest

import polars as pl
import pandera.polars as pa
from pandera.errors import SchemaError


@pytest.mark.parametrize(
    "dtype,data",
    [
        # python types
        (int, [1, 2, 3]),
        (str, ["foo", "bar", "baz"]),
        (float, [1.0, 2.0, 3.0]),
        (bool, [True, False, True]),
        # polars types
        (pl.Int64, [1, 2, 3]),
        (pl.Utf8, ["foo", "bar", "baz"]),
        (pl.Float64, [1.0, 2.0, 3.0]),
        (pl.Boolean, [True, False, True]),
    ],
)
def test_column_schema_simple_dtypes(dtype, data):
    schema = pa.Column(dtype, name="column")
    data = pl.DataFrame({"column": data}).lazy()
    validated_data = schema.validate(data).collect()
    assert validated_data.equals(data.collect())


@pytest.mark.parametrize(
    "column_kwargs",
    [
        {"name": r"^col_\d$", "regex": False},
        {"name": r"col_\d", "regex": True},
    ],
)
def test_column_schema_regex(column_kwargs):
    n_cols = 10
    schema = pa.Column(int, **column_kwargs)
    data = pl.DataFrame({f"col_{i}": [1, 2, 3] for i in range(n_cols)}).lazy()
    validated_data = data.pipe(schema.validate).collect()
    assert validated_data.equals(data.collect())

    for i in range(n_cols):
        invalid_data = data.cast({f"col_{i}": str})
        with pytest.raises(SchemaError):
            invalid_data.pipe(schema.validate).collect()


def test_get_regex_columns():

    ...


def test_coerce_dtype():
    ...


def test_check_nullable():
    ...


def test_check_unique():
    ...


def test_check_dtype():
    ...


def test_run_checks():
    ...


def test_set_default():
    ...
