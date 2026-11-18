"""Regression tests for validating the String-backed Category dtype."""

import datetime
from typing import Any, cast

import polars as pl
import pytest

import pandera.polars as pa
from pandera.engines.polars_engine import Category, String


@pytest.mark.parametrize("lazyframe", [False, True])
@pytest.mark.parametrize("coerce", [False, True])
def test_category_accepts_declared_values(
    lazyframe: bool, coerce: bool
) -> None:
    """Valid categories do not fail dtype validation after coercion."""
    schema = pa.DataFrameSchema(
        {
            "at": pa.Column(pl.Datetime),
            "kind": pa.Column(cast(Any, Category(categories=["a", "b", "c"]))),
        },
        coerce=coerce,
    )
    frame = pl.DataFrame(
        {"at": [datetime.datetime(2025, 1, 1)] * 2, "kind": ["a", "b"]}
    )
    if lazyframe:
        result = schema.validate(frame.lazy()).collect()
    else:
        result = schema.validate(frame)
    assert result.equals(frame)


@pytest.mark.parametrize("coerce", [False, True])
@pytest.mark.parametrize("values", [["a", None], []])
def test_category_nullable_and_empty(coerce: bool, values: list) -> None:
    """Nullability is validated separately, and empty strings are vacuous."""
    schema = pa.DataFrameSchema(
        {
            "kind": pa.Column(
                cast(Any, Category(categories=["a", "b"])), nullable=True
            )
        },
        coerce=coerce,
    )
    frame = pl.DataFrame({"kind": pl.Series(values, dtype=pl.String)})
    assert schema.validate(frame).equals(frame)


@pytest.mark.parametrize("coerce", [False, True])
def test_category_rejects_undeclared_values(coerce: bool) -> None:
    """Permitting the backing String dtype must still enforce categories."""
    schema = pa.DataFrameSchema(
        {"kind": pa.Column(cast(Any, Category(categories=["a", "b"])))},
        coerce=coerce,
    )
    with pytest.raises(pa.errors.SchemaErrors):
        schema.validate(pl.DataFrame({"kind": ["a", "outside"]}), lazy=True)


def test_category_dtype_comparison_remains_distinct() -> None:
    """A bare dtype comparison does not equate Category and String."""
    dtype = Category(categories=["a", "b"])
    assert not dtype.check(String())
    assert dtype.check(Category(categories=["a", "b"]))


def test_category_check_does_not_expand_unselected_container() -> None:
    """Whole frames and multi-column selectors retain dtype comparison."""
    from pandera.api.polars.types import PolarsData

    dtype = Category(categories=["a", "b"])
    frame = pl.LazyFrame({"kind1": ["a"], "kind2": ["b"]})
    assert not dtype.check(String(), PolarsData(frame))
    assert not dtype.check(String(), PolarsData(frame, "^kind.*$"))


@pytest.mark.parametrize("values", [[None], [1], ["outside"]])
def test_category_negative_values_still_fail(values: list) -> None:
    """Neither nullability nor unrelated value types bypass validation."""
    schema = pa.DataFrameSchema(
        {"kind": pa.Column(cast(Any, Category(categories=["a", "b"])))},
    )
    with pytest.raises(pa.errors.SchemaErrors):
        schema.validate(pl.DataFrame({"kind": values}), lazy=True)


def test_category_regex_columns() -> None:
    """Normal regex column validation selects each category column."""
    schema = pa.DataFrameSchema(
        {
            "^kind.*$": pa.Column(
                cast(Any, Category(categories=["a", "b"])), regex=True
            )
        }
    )
    frame = pl.DataFrame({"kind1": ["a"], "kind2": ["b"]})
    assert schema.validate(frame).equals(frame)
