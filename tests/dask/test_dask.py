""" Tests that basic Pandera functionality works for Dask objects. """

import dask.dataframe as dd
import pandas as pd
import pytest

import pandera as pa
from pandera.typing import DaskDataFrame, Series


class IntSchema(pa.SchemaModel):  # pylint: disable=missing-class-docstring
    col: Series[int]


class StrSchema(pa.SchemaModel):  # pylint: disable=missing-class-docstring
    col: Series[str]


def test_model_validation() -> None:
    """
    Test that model based pandera validation works with Dask DataFrames.
    """
    df = pd.DataFrame({"col": ["1"]})
    ddf = dd.from_pandas(df, npartitions=1)

    ddf = StrSchema.validate(ddf)
    pd.testing.assert_frame_equal(df, ddf.compute())

    ddf = IntSchema.validate(ddf)

    with pytest.raises(pa.errors.SchemaError):
        ddf.compute()


def test_dataframe_schema() -> None:
    """
    Test that DataFrameSchema based pandera validation works with Dask
    DataFrames.
    """
    int_schema = pa.DataFrameSchema({"col": pa.Column(int)})
    str_schema = pa.DataFrameSchema({"col": pa.Column(str)})

    df = pd.DataFrame({"col": ["1"]})
    ddf = dd.from_pandas(df, npartitions=1)

    ddf = str_schema.validate(ddf)
    pd.testing.assert_frame_equal(df, ddf.compute())

    ddf = int_schema.validate(ddf)

    with pytest.raises(pa.errors.SchemaError):
        ddf.compute()


def test_series_schema() -> None:
    """
    Test that SeriesSchema based pandera validation works with Dask Series.
    """
    integer_schema = pa.SeriesSchema(int)
    string_schema = pa.SeriesSchema(str)

    series = pd.Series(["1"])
    dseries = dd.from_pandas(series, npartitions=1)

    dseries = string_schema.validate(dseries)
    pd.testing.assert_series_equal(series, dseries.compute())

    dseries = integer_schema.validate(dseries)

    with pytest.raises(pa.errors.SchemaError):
        dseries.compute()


def test_decorator() -> None:
    """Test that pandera check_types decorator works with Dask DataFrames."""

    @pa.check_types
    def str_func(x: DaskDataFrame[StrSchema]) -> DaskDataFrame[StrSchema]:
        return x

    @pa.check_types
    def int_func(x: DaskDataFrame[IntSchema]) -> DaskDataFrame[IntSchema]:
        return x

    df = pd.DataFrame({"col": ["1"]})
    ddf = dd.from_pandas(df, npartitions=1)
    pd.testing.assert_frame_equal(df, str_func(ddf).compute())

    result = int_func(ddf)

    with pytest.raises(pa.errors.SchemaError):
        print(result.compute())
