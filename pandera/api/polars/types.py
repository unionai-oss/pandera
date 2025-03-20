"""Polars types."""

from typing import NamedTuple, Optional, Union, TypeVar

import polars as pl


class PolarsData(NamedTuple):
    lazyframe: pl.LazyFrame
    key: Optional[str] = None


class CheckResult(NamedTuple):
    """Check result for user-defined checks."""

    check_output: pl.LazyFrame
    check_passed: pl.LazyFrame
    checked_object: pl.LazyFrame
    failure_cases: pl.LazyFrame


PolarsCheckObjects = Union[pl.LazyFrame, pl.DataFrame]
PolarsFrame = TypeVar("PolarsFrame", pl.LazyFrame, pl.DataFrame)
PolarsFrame2 = TypeVar("PolarsFrame2", pl.LazyFrame, pl.DataFrame)

PolarsDtypeInputTypes = Union[
    str,
    type,
    pl.datatypes.classes.DataTypeClass,
]
