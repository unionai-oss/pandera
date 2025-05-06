"""Unit tests for polars LazyFrame generic."""

import polars as pl
import pytest

import pandera.polars as pa
from pandera.typing.polars import LazyFrame, Series


def test_series_annotation():
    class Model(pa.DataFrameModel):
        col1: Series[pl.Int64]

    data = pl.LazyFrame(
        {
            "col1": [1, 2, 3],
        }
    )

    assert data.collect().equals(Model.validate(data).collect())

    invalid_data = data.cast({"col1": pl.Float64})
    with pytest.raises(pa.errors.SchemaError):
        Model.validate(invalid_data).collect()


def test_lazyframe_generic_simple():
    class Model(pa.DataFrameModel):
        col1: pl.Int64
        col2: pl.Utf8
        col3: pl.Float64

    @pa.check_types
    def fn(lf: LazyFrame[Model]) -> LazyFrame[Model]:
        return lf

    data = pl.LazyFrame(
        {
            "col1": [1, 2, 3],
            "col2": [*"abc"],
            "col3": [1.0, 2.0, 3.0],
        }
    )

    assert data.collect().equals(fn(data).collect())

    invalid_data = data.cast({"col3": pl.Int64})
    with pytest.raises(pa.errors.SchemaError):
        fn(invalid_data).collect()


def test_lazyframe_generic_transform():
    class Input(pa.DataFrameModel):
        col1: pl.Int64
        col2: pl.Utf8

    class Output(Input):
        col3: pl.Float64

    @pa.check_types
    def fn(lf: LazyFrame[Input]) -> LazyFrame[Output]:
        return lf.with_columns(col3=pl.lit(3.0))  # type: ignore

    @pa.check_types
    def invalid_fn(lf: LazyFrame[Input]) -> LazyFrame[Output]:
        return lf  # type: ignore

    data = pl.LazyFrame(
        {
            "col1": [1, 2, 3],
            "col2": [*"abc"],
        }
    )

    assert isinstance(fn(data).collect(), pl.DataFrame)

    with pytest.raises(pa.errors.SchemaError):
        invalid_fn(data).collect()
