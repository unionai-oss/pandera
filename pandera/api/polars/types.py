"""Polars types."""

from typing import NamedTuple, Optional, Union

import polars as pl


class PolarsData(NamedTuple):
    dataframe: pl.LazyFrame
    key: Optional[str] = None


class CheckResult(NamedTuple):
    """Check result for user-defined checks."""

    check_output: pl.LazyFrame
    check_passed: pl.LazyFrame
    checked_object: pl.LazyFrame
    failure_cases: pl.LazyFrame


PolarsDtypeInputTypes = Union[
    str,
    type,
    pl.datatypes.classes.DataTypeClass,
]
