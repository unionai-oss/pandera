"""Polars types."""

from typing import NamedTuple, Optional, Union

from multimethod import parametric
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


PolarsDtypeInputTypes = Union[
    str,
    type,
    pl.datatypes.classes.DataTypeClass,
]


def is_bool(x):
    """Verifies whether an object is a boolean type."""
    return isinstance(x, (bool, pl.Boolean))


IsBool = parametric(object, is_bool)
