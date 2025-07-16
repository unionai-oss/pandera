"""Narwhals types."""

from typing import NamedTuple, Union, TypeVar, Any

import narwhals as nw


class NarwhalsData(NamedTuple):
    dataframe: nw.DataFrame[Any]
    key: str = "*"


class CheckResult(NamedTuple):
    """Check result for user-defined checks."""

    check_output: nw.DataFrame[Any]
    check_passed: nw.DataFrame[Any]
    checked_object: nw.DataFrame[Any]
    failure_cases: nw.DataFrame[Any]


NarwhalsCheckObjects = Union[nw.DataFrame[Any], nw.LazyFrame[Any]]
NarwhalsFrame = TypeVar("NarwhalsFrame", nw.DataFrame[Any], nw.LazyFrame[Any])

NarwhalsDtypeInputTypes = Union[
    str,
    type,
    nw.Dtype,
]
