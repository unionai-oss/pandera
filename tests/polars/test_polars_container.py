# pylint: disable=redefined-outer-name
"""Unit tests for polars container."""

import polars as pl

import pytest
import pandera as pa
from pandera import Check as C
from pandera.polars import Column, DataFrameSchema


@pytest.fixture
def ldf_basic():
    """Basic polars lazy dataframe fixture."""
    return pl.DataFrame(
        {"string_col": ["a", "b", "c"], "int_col": [0, 1, 2]}
    ).lazy()


@pytest.fixture
def ldf_schema_basic():
    """Basic polars lazyframe schema fixture."""
    return DataFrameSchema(
        {
            "string_col": Column(pl.Utf8),
            "int_col": Column(pl.Int64),
        }
    )


@pytest.fixture
def ldf_schema_with_check():
    """Polars lazyframe schema with checks."""
    return DataFrameSchema(
        {
            "string_col": Column(pl.Utf8),
            "int_col": Column(pl.Int64, C.ge(0)),
        }
    )


def test_basic_polars_lazy_dataframe(ldf_basic, ldf_schema_basic):
    """Test basic polars lazy dataframe."""
    query = ldf_basic.pipe(ldf_schema_basic.validate)
    df = query.collect()
    assert isinstance(query, pl.LazyFrame)
    assert isinstance(df, pl.DataFrame)


@pytest.mark.parametrize("lazy", [False, True])
def test_basic_polars_lazy_dataframe_dtype_error(
    lazy, ldf_basic, ldf_schema_basic
):
    """Test basic polars lazy dataframe."""
    query = ldf_basic.with_columns(pl.col("int_col").cast(pl.Int32))

    error_cls = pa.errors.SchemaErrors if lazy else pa.errors.SchemaError
    with pytest.raises(error_cls):
        # type check errors occur even before collection
        query.pipe(ldf_schema_basic.validate, lazy=lazy)


def test_basic_polars_lazy_dataframe_check_error(
    ldf_basic,
    ldf_schema_with_check,
):
    """Test basic polars lazy dataframe."""

    query = ldf_basic.pipe(ldf_schema_with_check.validate, lazy=True)

    validated_df = query.collect()
    assert validated_df.frame_equal(ldf_basic.collect())
