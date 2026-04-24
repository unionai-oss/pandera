"""Tests for :mod:`pandera.strategies.narwhals_strategies`.

Strategy generation for the Narwhals-backed schemas (polars + ibis) is
implemented by piggy-backing on ``pandera.strategies.pandas_strategies``:
generate a pandas DataFrame, then convert to the requested target.

These tests exercise:

- Direct :func:`narwhals_strategies.dataframe_strategy` calls for both
  polars and ibis targets.
- The container-level ``DataFrameSchema.example()`` /
  ``DataFrameSchema.strategy()`` shortcuts on the polars and ibis
  schemas.
- Built-in checks (``ge``, ``in_range``, ``isin``) survive the
  pandas-to-target conversion and the generated frame round-trips
  through ``schema.validate`` cleanly.
"""

from __future__ import annotations

import warnings

import pytest

pytest.importorskip("hypothesis")

import hypothesis  # noqa: E402
import polars as pl  # noqa: E402

import pandera.polars as pa_pl  # noqa: E402
from pandera.api.checks import Check  # noqa: E402
from pandera.strategies import narwhals_strategies as nws  # noqa: E402


@pytest.fixture(autouse=True)
def _silence_noninteractive_example_warning():
    """``.example()`` emits a noisy warning that's irrelevant for these tests."""
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore", category=hypothesis.errors.NonInteractiveExampleWarning
        )
        yield


# ---------------------------------------------------------------------------
# dtype translation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "polars_dtype,expected",
    [
        (pl.Int8, "int8"),
        (pl.Int64, "int64"),
        (pl.UInt32, "uint32"),
        (pl.Float64, "float64"),
        (pl.Boolean, "bool"),
        (pl.String, "object"),
    ],
)
def test_polars_dtype_to_numpy(polars_dtype, expected):
    pd_dtype = nws._polars_dtype_to_numpy(polars_dtype)
    assert str(pd_dtype) == expected


def test_ibis_dtype_to_numpy():
    dt = pytest.importorskip("ibis.expr.datatypes")
    assert str(nws._ibis_dtype_to_numpy(dt.int64)) == "int64"
    assert str(nws._ibis_dtype_to_numpy(dt.float64)) == "float64"
    assert str(nws._ibis_dtype_to_numpy(dt.boolean)) == "bool"


# ---------------------------------------------------------------------------
# Polars DataFrameSchema strategy / example
# ---------------------------------------------------------------------------


def test_polars_schema_example_returns_polars_dataframe():
    schema = pa_pl.DataFrameSchema(
        {
            "a": pa_pl.Column(pl.Int64, checks=Check.ge(0)),
            "b": pa_pl.Column(pl.String),
            "c": pa_pl.Column(pl.Float64, nullable=True),
        }
    )
    out = schema.example(size=5)
    assert isinstance(out, pl.DataFrame)
    assert out.height == 5
    schema.validate(out)


def test_polars_schema_example_respects_in_range():
    schema = pa_pl.DataFrameSchema(
        {"x": pa_pl.Column(pl.Int64, checks=Check.in_range(10, 20))}
    )
    out = schema.example(size=10)
    assert isinstance(out, pl.DataFrame)
    xs = out["x"].to_list()
    assert all(10 <= v <= 20 for v in xs)


def test_polars_schema_example_respects_isin():
    schema = pa_pl.DataFrameSchema(
        {"g": pa_pl.Column(pl.String, checks=Check.isin(["a", "b", "c"]))}
    )
    out = schema.example(size=10)
    assert set(out["g"].to_list()) <= {"a", "b", "c"}


def test_polars_schema_strategy_can_draw():
    """``schema.strategy()`` returns a hypothesis SearchStrategy."""
    from hypothesis.strategies import SearchStrategy

    schema = pa_pl.DataFrameSchema(
        {"a": pa_pl.Column(pl.Int64, checks=Check.ge(0))}
    )
    strategy = schema.strategy(size=4)
    assert isinstance(strategy, SearchStrategy)
    drawn = strategy.example()
    assert isinstance(drawn, pl.DataFrame)
    schema.validate(drawn)


# ---------------------------------------------------------------------------
# Polars Column strategy / example
# ---------------------------------------------------------------------------


def test_polars_column_example_returns_dataframe():
    col = pa_pl.Column(pl.Int64, name="a", checks=Check.ge(5))
    out = col.example(size=4)
    assert isinstance(out, pl.DataFrame)
    assert "a" in out.columns
    assert all(v >= 5 for v in out["a"].to_list())


# ---------------------------------------------------------------------------
# Ibis DataFrameSchema strategy / example
# ---------------------------------------------------------------------------


def test_ibis_schema_example_returns_ibis_table():
    ibis = pytest.importorskip("ibis")
    dt = pytest.importorskip("ibis.expr.datatypes")

    import pandera.ibis as pa_ib

    ibis.set_backend("duckdb")

    schema = pa_ib.DataFrameSchema(
        {
            "a": pa_ib.Column(dt.int64, checks=Check.ge(0)),
            "b": pa_ib.Column(dt.string),
        }
    )
    out = schema.example(size=4)
    assert isinstance(out, ibis.Table)
    df = out.execute()
    assert df.shape == (4, 2)
    assert all(v >= 0 for v in df["a"].tolist())
    schema.validate(out)


def test_ibis_schema_strategy_can_draw():
    ibis = pytest.importorskip("ibis")
    dt = pytest.importorskip("ibis.expr.datatypes")

    from hypothesis.strategies import SearchStrategy

    import pandera.ibis as pa_ib

    ibis.set_backend("duckdb")

    schema = pa_ib.DataFrameSchema(
        {"x": pa_ib.Column(dt.int64, checks=Check.in_range(10, 20))}
    )
    strategy = schema.strategy(size=6)
    assert isinstance(strategy, SearchStrategy)
    drawn = strategy.example()
    df = drawn.execute()
    assert all(10 <= v <= 20 for v in df["x"].tolist())


# ---------------------------------------------------------------------------
# Lazy target (polars LazyFrame)
# ---------------------------------------------------------------------------


def test_dataframe_strategy_polars_lazy_target():
    """The ``polars_lazy`` target wraps the result with ``.lazy()``."""
    schema = pa_pl.DataFrameSchema(
        {"v": pa_pl.Column(pl.Int64, checks=Check.ge(0))}
    )

    strategy = nws.dataframe_strategy(
        columns=schema.columns,
        size=3,
        target="polars_lazy",
    )
    drawn = strategy.example()
    assert isinstance(drawn, pl.LazyFrame)
    collected = drawn.collect()
    assert collected.height == 3


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_dataframe_strategy_unknown_target_rejected():
    schema = pa_pl.DataFrameSchema({"a": pa_pl.Column(pl.Int64)})
    strategy = nws.dataframe_strategy(
        columns=schema.columns,
        size=1,
        target="not-a-real-target",
    )
    with pytest.raises(ValueError, match="Unknown target"):
        strategy.example()


def test_to_pandas_dtype_returns_none_for_none():
    """``_to_pandas_dtype(None)`` short-circuits to ``None``."""
    assert nws._to_pandas_dtype(None) is None


def test_polars_dtype_unknown_raises_not_implemented():
    """A bogus polars dtype object surfaces a clear NotImplementedError."""

    class _BadDtype:
        __module__ = "polars.datatypes._fake"

    with pytest.raises(NotImplementedError, match="polars dtype"):
        nws._polars_dtype_to_numpy(_BadDtype)
