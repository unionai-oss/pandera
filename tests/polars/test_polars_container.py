"""Unit tests for polars container."""

import polars as pl

import pytest
import pandera as pa
from pandera import Check as C
from pandera.polars import Column, DataFrameSchema


@pytest.fixture
def ldf_basic():
    return pl.DataFrame(
        {"string_col": ["a", "b", "c"], "int_col": [0, 1, 2]}
    ).lazy()


@pytest.fixture
def ldf_schema_basic():
    return DataFrameSchema(
        {
            "string_col": Column(pl.Utf8),
            "int_col": Column(pl.Int64),
        }
    )


@pytest.fixture
def ldf_schema_with_check():
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

    # TODO:
    # By definition pandera needs to do non-lazy operations on the data to
    # to the run-time value checks. Pandera can run metadata checks, e.g.
    # data type checks, column name uniqueness, etc.
    #
    # This is because the LazyFrame API propagates type information
    # through a lazy query, but it cannot do run-time value checks without
    # materializing the data at validation time.
    #
    # Therefore, checks that require examining the values of the data to raise
    # an error will do a series of non-lazy operations on the data, ideally in
    # parallel, before raising a runtime error on collect.
    #
    # Calling schema.validate should run an implicit collect(), and may also
    # do an implicit `lazy()` to continue the lazy operations.
    #
    # Idea: we formalize two modes of validation:
    # 1. Metadata validation: check metadata such as primitive datatypes,
    #    e.g. int64, string, etc.
    # 2. Data value validation: check actual values.
    #
    # In the polars programming model, we can do metadata validation before even
    # running the query, but we need to actually run the query to gather the
    # failure cases for data values that don't pass run-time checks
    # (e.g. col >= 0).
    #
    # In order to lazily raise a data value error, pandera can introduce a
    # namespace:
    #
    # (
    #    ldf
    #    .pandera.validate(schema, collect=False)  # raises metadata errors
    #    .with_columns(...)  # do stuff
    #    .pandera.collect()  # this runs the query, raising a data value error.
    #                        # collect() also materializes a pl.DataFrame
    #    .lazy()             # convert back to lazy as desired
    # )
    #
    # Supporting this would require adding support for lazy evaluation of
    # checks, so instead of `CoreCheckResult` and `CheckResult`, it would
    # require a `CoreCheckPromise`,  `CheckPromise`, which would contain
    # LazyFrames or some other promise of an actual result. These would then
    # be run by calling `polars.collect_all()` when `pandera.collect` is
    # invoked.

    query = ldf_basic.pipe(ldf_schema_with_check.validate, lazy=True)

    validated_df = query.collect()
    validated_df == ldf_basic.collect()
    assert validated_df.frame_equal(ldf_basic.collect())
