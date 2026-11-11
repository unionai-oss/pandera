"""Unit tests for polars strategy methods."""

import polars as pl
import pytest

import pandera.polars as pa


def test_dataframe_schema_strategy():
    schema = pa.DataFrameSchema({"a": pa.Column(int, pa.Check.ge(0))})

    # Data synthesis for polars schemas is supported via the shared value
    # core; ``strategy()`` returns a builder whose ``.example()`` assembles a
    # native ``pl.DataFrame`` (see tests/strategies/test_synthesis_polars.py).
    df = schema.strategy(size=3).example()
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 3

    assert isinstance(schema.example(size=3), pl.DataFrame)


def test_column_schema_strategy():
    column_schema = pa.Column(str)

    with pytest.raises(NotImplementedError):
        column_schema.strategy()

    with pytest.raises(NotImplementedError):
        column_schema.example()

    with pytest.raises(NotImplementedError):
        column_schema.strategy_component()
