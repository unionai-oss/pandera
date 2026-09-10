"""PyArrow types."""

from typing import NamedTuple, Union

import pyarrow as pa


class PyArrowData(NamedTuple):
    """Data container passed to ``native=True`` pyarrow checks.

    Mirrors :class:`~pandera.api.polars.types.PolarsData` and
    :class:`~pandera.api.ibis.types.IbisData` so that check functions taking a
    single positional argument receive the same shape across backends.
    """

    table: pa.Table
    key: str = "*"


class CheckResult(NamedTuple):
    """Check result for user-defined checks."""

    check_output: pa.Table
    check_passed: pa.Table
    checked_object: pa.Table
    failure_cases: pa.Table


PyArrowCheckObjects = pa.Table

PyArrowDtypeInputTypes = Union[
    str,
    type,
    pa.DataType,
]
