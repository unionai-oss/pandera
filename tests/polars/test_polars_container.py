"""Unit tests for polars container."""

import polars as pl

import pytest
import pandera as pa
from pandera.polars import Column, DataFrameSchema


@pytest.fixture
def basic_ldf():
    return pl.DataFrame(
        {"string_col": ["a", "b", "c"], "int_col": [0, 1, 2]}
    ).lazy()


@pytest.fixture
def basic_ldf_schema():
    return DataFrameSchema(
        {
            "string_col": Column(pl.Utf8),
            "int_col": Column(pl.Int64),
        }
    )


def test_basic_polars_lazy_dataframe(basic_ldf, basic_ldf_schema):
    """Test basic polars lazy dataframe."""
    query = basic_ldf.pipe(basic_ldf_schema.validate)
    df = query.collect()
    assert isinstance(query, pl.LazyFrame)
    assert isinstance(df, pl.DataFrame)


@pytest.mark.parametrize("lazy", [False, True])
def test_basic_polars_lazy_dataframe_dtype_error(
    lazy, basic_ldf, basic_ldf_schema
):
    """Test basic polars lazy dataframe."""
    query = basic_ldf.with_columns(pl.col("int_col").cast(pl.Int32))

    error_cls = pa.errors.SchemaErrors if lazy else pa.errors.SchemaError
    with pytest.raises(error_cls):
        # type check errors occur even before collection
        query.pipe(basic_ldf_schema.validate, lazy=lazy)
