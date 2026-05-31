"""Tests for cross-backend check signature compatibility under Narwhals.

These tests pin the contract introduced in ``fix/narwhals-incompat``:
polars-style (``def fn(data: PolarsData)``) and ibis-style (``def
fn(data: IbisData)``) user check functions — whether defined directly
via ``pa.Check(fn)`` or via the ``@pa.check`` model decorator — must
work under the Narwhals backend just as they do under the native
polars / ibis backends.

The Narwhals backend already supports the 2-positional ``def fn(frame,
key)`` convention (see ``tests/narwhals/test_checks.py``). This module
guards the *additional* 1-positional convention so it never regresses.
"""

from __future__ import annotations

import pandas as pd
import polars as pl
import pytest

import pandera.polars as pa_polars
from pandera.api.polars.types import PolarsData

try:
    import ibis as _ibis_mod  # noqa: F401

    HAS_IBIS = True
except ImportError:
    HAS_IBIS = False

ibis_only = pytest.mark.skipif(not HAS_IBIS, reason="ibis not installed")


@pytest.fixture
def polars_invalid_df() -> pl.DataFrame:
    """A polars DataFrame with one negative value in ``col``."""
    return pl.DataFrame({"col": [1, 2, -3]})


@pytest.fixture
def ibis_invalid_table():
    """An ibis memtable mirroring ``polars_invalid_df``."""
    import ibis

    return ibis.memtable(pd.DataFrame({"col": [1, 2, -3]}))


# ---------------------------------------------------------------------------
# Direct ``pa.Check(fn)`` invocation — polars/ibis-style 1-positional fns
# ---------------------------------------------------------------------------


def test_direct_polars_style_check_receives_polars_data(polars_invalid_df):
    """``pa.Check(fn)`` with a ``def fn(data: PolarsData)`` function must
    receive a ``PolarsData`` (not the legacy 2-positional ``(frame, key)``)
    when dispatched through the Narwhals backend."""

    received: list[PolarsData] = []

    def my_check(data: PolarsData) -> pl.LazyFrame:
        received.append(data)
        return data.lazyframe.select(pl.col(data.key).ge(0))

    schema = pa_polars.DataFrameSchema(
        {"col": pa_polars.Column(int, checks=pa_polars.Check(my_check))}
    )

    with pytest.raises(pa_polars.errors.SchemaError):
        schema.validate(polars_invalid_df)

    assert len(received) == 1
    data = received[0]
    assert isinstance(data, PolarsData)
    assert data.key == "col"
    assert isinstance(data.lazyframe, pl.LazyFrame)


@ibis_only
def test_direct_ibis_style_check_receives_ibis_data(ibis_invalid_table):
    """``pa.Check(fn)`` with a ``def fn(data: IbisData)`` function must
    receive an ``IbisData`` (not ``(frame, key)``) under the Narwhals
    backend."""
    import ibis

    import pandera.ibis as pa_ibis
    from pandera.api.ibis.types import IbisData

    received: list[IbisData] = []

    def my_check(data: IbisData):
        received.append(data)
        return data.table[data.key] >= 0

    schema = pa_ibis.DataFrameSchema(
        {"col": pa_ibis.Column(int, checks=pa_ibis.Check(my_check))}
    )

    with pytest.raises(pa_ibis.errors.SchemaError):
        schema.validate(ibis_invalid_table)

    assert len(received) == 1
    data = received[0]
    assert isinstance(data, IbisData)
    assert data.key == "col"
    assert isinstance(data.table, ibis.Table)


# ---------------------------------------------------------------------------
# ``@pa.check`` model decorator — exercises the ``_adapter`` 1-positional path
# ---------------------------------------------------------------------------


def test_model_check_polars_receives_polars_data(polars_invalid_df):
    """``@pa.check`` on a ``DataFrameModel`` with a ``PolarsData`` parameter
    must run end-to-end and capture the failing row in ``failure_cases``."""

    class S(pa_polars.DataFrameModel):
        col: int

        @pa_polars.check("col", name="non_negative")
        @classmethod
        def non_negative(cls, data: PolarsData) -> pl.LazyFrame:  # noqa: ARG002
            return data.lazyframe.select(pl.col(data.key).ge(0))

    with pytest.raises(pa_polars.errors.SchemaErrors) as exc_info:
        S.validate(polars_invalid_df, lazy=True)

    failure_cases = exc_info.value.failure_cases
    assert "non_negative" in failure_cases["check"].to_list()
    rendered = failure_cases["failure_case"].to_list()
    assert any("-3" in str(v) for v in rendered)


@ibis_only
def test_model_check_ibis_receives_ibis_data(ibis_invalid_table):
    """``@pa.check`` on a ``DataFrameModel`` with an ``IbisData`` parameter
    must run end-to-end against an ibis backend table."""
    import pandera.ibis as pa_ibis
    from pandera.api.ibis.types import IbisData

    class S(pa_ibis.DataFrameModel):
        col: int

        @pa_ibis.check("col", name="non_negative")
        @classmethod
        def non_negative(cls, data: IbisData):  # noqa: ARG002
            return data.table[data.key] >= 0

    with pytest.raises(pa_ibis.errors.SchemaErrors) as exc_info:
        S.validate(ibis_invalid_table, lazy=True)

    failure_cases = exc_info.value.failure_cases
    # ibis renders integer failure_cases as ints, polars as strings — accept
    # both forms so the assertion stays backend-agnostic.
    rendered = [str(v) for v in failure_cases["failure_case"].to_list()]
    assert any("-3" in v for v in rendered)


# ---------------------------------------------------------------------------
# Output normalization — polars-style checks may return any of:
#   pl.Series, pl.DataFrame, pl.LazyFrame; with one or many bool cols.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "check_fn",
    [
        # Single-column LazyFrame (most common polars-style return).
        lambda data: data.lazyframe.select(pl.col(data.key).ge(0)),
        # Single-column DataFrame (collected eagerly inside the check).
        lambda data: data.lazyframe.select(pl.col(data.key).ge(0)).collect(),
        # Multi-column boolean LazyFrame — must AND-reduce.
        lambda data: data.lazyframe.select(
            pl.col(data.key).ge(0).alias("ge_zero"),
            pl.col(data.key).is_not_null().alias("not_null"),
        ),
        # pl.Series return.
        lambda data: data.lazyframe.collect()[data.key] >= 0,
    ],
    ids=["lazyframe_1col", "dataframe_1col", "lazyframe_multicol", "series"],
)
def test_polars_check_output_shape_normalization(polars_invalid_df, check_fn):
    """Various polars-style return shapes all reduce to a single
    boolean check-output column under the Narwhals backend."""

    schema = pa_polars.DataFrameSchema(
        {"col": pa_polars.Column(int, checks=pa_polars.Check(check_fn))}
    )

    # lazy=True produces a normalised ``failure_cases`` table with a
    # ``failure_case`` column; without lazy, ``failure_cases`` mirrors
    # the original frame.
    with pytest.raises(pa_polars.errors.SchemaErrors) as exc_info:
        schema.validate(polars_invalid_df, lazy=True)

    # The failing row's value (-3) must surface in the rendered failure
    # case, regardless of the return-shape variant the check used.
    failure_case = exc_info.value.failure_cases["failure_case"].to_list()
    assert any("-3" in str(v) for v in failure_case)


# ---------------------------------------------------------------------------
# Regression guard for the original 2-positional ``def fn(frame, key)``
# narwhals-native convention. This test ensures the new arity-based
# dispatch does not break the legacy contract.
# ---------------------------------------------------------------------------


def test_two_arg_narwhals_native_convention_preserved(polars_invalid_df):
    """``pa.Check(def fn(frame, key))`` (legacy narwhals-native signature)
    must continue to receive a *raw* native frame and the column name."""

    received: list[tuple[type, str]] = []

    def my_check(frame, key):
        received.append((type(frame), key))
        return frame.select(pl.col(key).ge(0))

    schema = pa_polars.DataFrameSchema(
        {"col": pa_polars.Column(int, checks=pa_polars.Check(my_check))}
    )

    with pytest.raises(pa_polars.errors.SchemaError):
        schema.validate(polars_invalid_df)

    assert len(received) == 1
    frame_type, key = received[0]
    # The native frame is a polars frame (LazyFrame or DataFrame), *not*
    # a PolarsData wrapper — that's the whole point of this contract.
    assert "polars" in frame_type.__module__
    assert key == "col"
