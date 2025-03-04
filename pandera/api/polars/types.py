"""Polars types."""

from typing import NamedTuple, Union

import polars as pl


class AllColumnsPolarsCheckData(NamedTuple):
    """Container for check data applying to all columns."""

    lazyframe: pl.LazyFrame

    @property
    def key(self) -> None:
        """Dummy attribute to allow deducing all columns check data from single columns without isinstance checks."""
        return None


class PolarsData(NamedTuple):
    """Container for check data which applies to a single column."""

    lazyframe: pl.LazyFrame
    key: str


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
