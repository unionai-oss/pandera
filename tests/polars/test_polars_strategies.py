"""Unit tests for polars strategy methods."""

import pytest

import pandera.polars as pa


def test_dataframe_schema_strategy():
    schema = pa.DataFrameSchema()

    with pytest.raises(NotImplementedError):
        schema.strategy()

    with pytest.raises(NotImplementedError):
        schema.example()


def test_column_schema_strategy():
    column_schema = pa.Column(str)

    with pytest.raises(NotImplementedError):
        column_schema.strategy()

    with pytest.raises(NotImplementedError):
        column_schema.example()

    with pytest.raises(NotImplementedError):
        column_schema.strategy_component()
