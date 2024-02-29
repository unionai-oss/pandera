# pylint: disable=redefined-outer-name
"""Unit tests for polars container."""

import polars as pl

import pytest
import pandera as pa
from pandera import Check as C
from pandera.api.polars.types import PolarsData
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
    assert validated_df.equals(ldf_basic.collect())


def test_coerce_column_dtype(ldf_basic, ldf_schema_basic):
    """Test coerce dtype via column-level dtype specification."""
    ldf_schema_basic._coerce = True
    modified_data = ldf_basic.with_columns(pl.col("int_col").cast(pl.Utf8))
    query = modified_data.pipe(ldf_schema_basic.validate)
    coerced_df = query.collect()
    assert coerced_df.equals(ldf_basic.collect())


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
    assert coerced_df.equals(ldf_basic.cast(pl.Utf8).collect())


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
    assert validated_data.collect().equals(modified_data.collect())

    # setting strict to True should raise an error
    ldf_schema_basic.strict = True
    with pytest.raises(pa.errors.SchemaError):
        modified_data.pipe(ldf_schema_basic.validate)

    # setting strict to "filter" should remove the extra column
    ldf_schema_basic.strict = "filter"
    filtered_data = modified_data.pipe(ldf_schema_basic.validate)
    filtered_data.collect().equals(ldf_basic.collect())


def test_add_missing_columns_with_default(ldf_basic, ldf_schema_basic):
    """Test add_missing_columns argument with a default value."""
    ldf_schema_basic.add_missing_columns = True
    ldf_schema_basic.columns["int_col"].default = 1
    modified_data = ldf_basic.drop("int_col")
    validated_data = modified_data.pipe(ldf_schema_basic.validate)
    assert validated_data.collect().equals(
        ldf_basic.with_columns(int_col=pl.lit(1)).collect()
    )


def test_add_missing_columns_with_nullable(ldf_basic, ldf_schema_basic):
    """Test add_missing_columns argument with a nullable value."""
    ldf_schema_basic.add_missing_columns = True
    ldf_schema_basic.columns["int_col"].nullable = True
    modified_data = ldf_basic.drop("int_col")
    validated_data = modified_data.pipe(ldf_schema_basic.validate)
    assert validated_data.collect().equals(
        ldf_basic.with_columns(int_col=pl.lit(None)).collect()
    )


def test_unique_column_names(ldf_basic, ldf_schema_basic):
    """Test unique column names."""
    ldf_schema_basic.unique_column_names = True
    with pytest.warns():
        ldf_basic.pipe(ldf_schema_basic.validate).collect()


def test_column_absent_error(ldf_basic, ldf_schema_basic):
    """Test column presence."""
    with pytest.raises(
        pa.errors.SchemaError, match="column 'int_col' not in dataframe"
    ):
        ldf_basic.drop("int_col").pipe(ldf_schema_basic.validate).collect()


def test_column_values_are_unique(ldf_basic, ldf_schema_basic):
    """Test column values are unique."""
    ldf_schema_basic.unique = ["string_col", "int_col"]
    modified_data = ldf_basic.with_columns(
        string_col=pl.lit("a"), int_col=pl.lit(0)
    )
    with pytest.raises(pa.errors.SchemaError):
        modified_data.pipe(ldf_schema_basic.validate).collect()


def test_dataframe_level_checks():
    def custom_check(data: PolarsData):
        return data.dataframe.select(pl.col("*").eq(0))

    schema = DataFrameSchema(
        columns={"a": Column(pl.Int64), "b": Column(pl.Int64)},
        checks=[
            pa.Check(custom_check),
            pa.Check(lambda d: d.dataframe.select(pl.col("*").eq(0))),
        ],
    )
    ldf = pl.DataFrame({"a": [0, 0, 1, 1], "b": [0, 1, 0, 1]}).lazy()
    with pytest.raises(pa.errors.SchemaError):
        ldf.pipe(schema.validate)

    try:
        ldf.pipe(schema.validate, lazy=True)
    except pa.errors.SchemaErrors as err:
        assert err.failure_cases.shape[0] == 6
