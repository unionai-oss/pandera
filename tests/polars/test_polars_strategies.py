"""Unit tests for polars strategy methods.

These previously asserted that ``strategy()`` and ``example()`` raised
``NotImplementedError``. The narwhals backend now implements them by
delegating to the pandas strategies and converting the result to polars.
"""

import warnings

import polars as pl
import pytest

import pandera.polars as pa

pytest.importorskip("hypothesis")
import hypothesis  # noqa: E402


@pytest.fixture(autouse=True)
def _silence_noninteractive_example_warning():
    """``.example()`` emits a noisy warning that is irrelevant for these tests."""
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            category=hypothesis.errors.NonInteractiveExampleWarning,
        )
        yield


def test_dataframe_schema_strategy_emits_polars_dataframe():
    schema = pa.DataFrameSchema({"a": pa.Column(pl.Int64)})
    out = schema.example(size=2)
    assert isinstance(out, pl.DataFrame)
    assert out.height == 2

    strategy = schema.strategy(size=2)
    drawn = strategy.example()
    assert isinstance(drawn, pl.DataFrame)


def test_column_schema_strategy_emits_polars_dataframe():
    column = pa.Column(pl.Int64, name="x")
    out = column.example(size=3)
    assert isinstance(out, pl.DataFrame)
    assert out.height == 3
    assert "x" in out.columns


def test_column_schema_strategy_component_still_unimplemented():
    """``strategy_component`` is not part of the polars backend surface."""
    column = pa.Column(pl.Int64, name="x")
    with pytest.raises(NotImplementedError):
        column.strategy_component()
