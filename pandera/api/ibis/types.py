"""Ibis types."""

from typing import NamedTuple, Optional, Union

import ibis
import ibis.expr.datatypes as dt


class IbisData(NamedTuple):
    table: ibis.Table
    key: str | None = None

    @property
    def frame(self) -> ibis.Table:
        """The native dataframe (Table)."""
        return self.table


class CheckResult(NamedTuple):
    """Check result for user-defined checks."""

    check_output: ibis.Table
    check_passed: ibis.Table
    checked_object: ibis.Table
    failure_cases: ibis.Table


IbisCheckObjects = Union[ibis.Table, ibis.Column]

IbisDtypeInputTypes = Union[
    str,
    type,
    dt.DataType,
]
