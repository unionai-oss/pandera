"""Polars types."""

from typing import NamedTuple, TypeVar, Union

import polars as pl
from polars.datatypes.classes import DataType, DataTypeClass


class PolarsData(NamedTuple):
    lazyframe: pl.LazyFrame
    key: str = "*"


class CheckResult(NamedTuple):
    """Check result for user-defined checks."""

    check_output: pl.LazyFrame
    check_passed: pl.LazyFrame
    checked_object: pl.LazyFrame
    failure_cases: pl.LazyFrame


PolarsCheckObjects = Union[pl.LazyFrame, pl.DataFrame]
PolarsFrame = TypeVar("PolarsFrame", pl.LazyFrame, pl.DataFrame)

PolarsDtypeInputTypes = Union[
    str,
    type,
    DataTypeClass,
    DataType,
]
