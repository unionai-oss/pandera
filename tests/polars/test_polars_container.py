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


def test_coerce_column_dtype(ldf_basic, ldf_schema_basic):
    """Test coerce dtype via column-level dtype specification."""
    ldf_schema_basic._coerce = True
    modified_data = ldf_basic.with_columns(pl.col("int_col").cast(pl.Utf8))
    query = modified_data.pipe(ldf_schema_basic.validate)
    coerced_df = query.collect()
    assert coerced_df.frame_equal(ldf_basic.collect())


def test_coerce_column_dtype_error(ldf_basic, ldf_schema_basic):
    """Test coerce dtype raises error when values cannot be coerced."""
    ldf_schema_basic._coerce = True

    # change dtype of strong_col to int64, where coercion of values should fail
    ldf_schema_basic.columns["string_col"].dtype = pl.Int64
    with pytest.raises(pa.errors.SchemaError):
        ldf_basic.pipe(ldf_schema_basic.validate)


def test_coerce_df_dtype(ldf_basic, ldf_schema_basic):
    """Test coerce dtype via dataframe-level dtype specification."""
    ldf_schema_basic._coerce = True
    ldf_schema_basic.dtype = pl.Utf8
    ldf_schema_basic.columns["int_col"].dtype = pl.Utf8
    query = ldf_basic.pipe(ldf_schema_basic.validate)
    coerced_df = query.collect()
    assert coerced_df.frame_equal(ldf_basic.cast(pl.Utf8).collect())


def test_coerce_df_dtype_error(ldf_basic, ldf_schema_basic):
    """Test coerce dtype when values cannot be coerced."""
    ldf_schema_basic._coerce = True

    # change dtype of schema to int64, where string_col value coercion should
    # fail
    ldf_schema_basic.dtype = pl.Int64
    ldf_schema_basic.columns["string_col"].dtype = pl.Int64
    with pytest.raises(pa.errors.SchemaError):
        ldf_basic.pipe(ldf_schema_basic.validate)


def test_strict_filter(ldf_basic, ldf_schema_basic):
    """Test strictness and filtering schema logic."""
    # by default, strict is False, so by default it should pass
    modified_data = ldf_basic.with_columns(extra_col=pl.lit(1))
    validated_data = modified_data.pipe(ldf_schema_basic.validate)
    assert validated_data.collect().frame_equal(modified_data.collect())

    # setting strict to True should raise an error
    ldf_schema_basic.strict = True
    with pytest.raises(pa.errors.SchemaError):
        modified_data.pipe(ldf_schema_basic.validate)

    # setting strict to "filter" should remove the extra column
    ldf_schema_basic.strict = "filter"
    filtered_data = modified_data.pipe(ldf_schema_basic.validate)
    filtered_data.collect().frame_equal(ldf_basic.collect())
