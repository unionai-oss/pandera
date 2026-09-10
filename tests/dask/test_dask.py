"""Tests that basic Pandera functionality works for Dask objects."""

from typing import cast

import dask.dataframe as dd
import pandas as pd
import pytest

import pandera.pandas as pa
from pandera.typing.dask import DataFrame, Series


class IntSchema(pa.DataFrameModel):  # pylint: disable=missing-class-docstring
    col: Series[int]


class StrSchema(pa.DataFrameModel):  # pylint: disable=missing-class-docstring
    col: Series[pd.StringDtype] = pa.Field(dtype_kwargs={"storage": "pyarrow"})


def test_model_validation() -> None:
    """
    Test that model based pandera validation works with Dask DataFrames.
    """
    df = pd.DataFrame(
        {"col": pd.Series(["1"], dtype=pd.StringDtype(storage="pyarrow"))}
    )
    ddf = dd.from_pandas(df, npartitions=1)

    ddf = StrSchema.validate(ddf)  # type: ignore[arg-type]
    pd.testing.assert_frame_equal(df, ddf.compute())  # type: ignore [attr-defined]

    ddf = IntSchema.validate(ddf)  # type: ignore[arg-type]

    with pytest.raises(pa.errors.SchemaError):
        ddf.compute()  # type: ignore [attr-defined]

    IntSchema.validate(ddf, inplace=True)  # type: ignore[arg-type]

    with pytest.raises(pa.errors.SchemaError):
        ddf.compute()  # type: ignore [attr-defined]


def test_dataframe_schema() -> None:
    """
    Test that DataFrameSchema based pandera validation works with Dask
    DataFrames.
    """
    int_schema = pa.DataFrameSchema({"col": pa.Column(int)})
    str_schema = pa.DataFrameSchema({"col": pa.Column(str)})

    df = pd.DataFrame(
        {"col": pd.Series(["1"], dtype=pd.StringDtype(storage="pyarrow"))}
    )
    ddf = dd.from_pandas(df, npartitions=1)

    ddf = str_schema.validate(ddf)  # type: ignore[arg-type]
    pd.testing.assert_frame_equal(df, ddf.compute())  # type: ignore[operator]

    ddf = int_schema.validate(ddf)

    with pytest.raises(pa.errors.SchemaError):
        ddf.compute()  # type: ignore[operator]

    IntSchema.validate(ddf, inplace=True)

    with pytest.raises(pa.errors.SchemaError):
        ddf.compute()  # type: ignore[operator]


def test_series_schema() -> None:
    """
    Test that SeriesSchema based pandera validation works with Dask Series.
    """
    integer_schema = pa.SeriesSchema(int)
    string_schema = pa.SeriesSchema(str)

    series = pd.Series(["1"], dtype=pd.StringDtype(storage="pyarrow"))
    dseries = dd.from_pandas(series, npartitions=1)

    dseries = string_schema.validate(dseries)  # type: ignore[arg-type]
    pd.testing.assert_series_equal(series, dseries.compute())

    dseries = integer_schema.validate(dseries)

    with pytest.raises(pa.errors.SchemaError):
        dseries.compute()

    integer_schema.validate(dseries, inplace=True)

    with pytest.raises(pa.errors.SchemaError):
        dseries.compute()


def test_decorator() -> None:
    """Test that pandera check_types decorator works with Dask DataFrames."""

    @pa.check_types
    def str_func(x: DataFrame[StrSchema]) -> DataFrame[StrSchema]:
        return x

    @pa.check_types
    def int_func(x: DataFrame[IntSchema]) -> DataFrame[IntSchema]:
        return x

    df = pd.DataFrame(
        {"col": pd.Series(["1"], dtype=pd.StringDtype(storage="pyarrow"))}
    )
    ddf = dd.from_pandas(df, npartitions=1)
    pd.testing.assert_frame_equal(
        df, str_func(cast(pa.typing.dask.DataFrame[StrSchema], ddf)).compute()
    )

    result = int_func(cast(pa.typing.dask.DataFrame[IntSchema], ddf))

    with pytest.raises(pa.errors.SchemaError):
        print(result.compute())
