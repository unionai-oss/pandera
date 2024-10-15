"""Ibis types."""

from typing import NamedTuple, Optional, Union

import ibis.expr.datatypes as dt
import ibis.expr.types as ir


class IbisData(NamedTuple):
    table: ir.Table
    key: Optional[str] = None


class CheckResult(NamedTuple):
    """Check result for user-defined checks."""

    check_output: ir.Table
    check_passed: ir.Table
    checked_object: ir.Table
    failure_cases: ir.Table


IbisCheckObjects = Union[ir.Table, ir.Column]


IbisDtypeInputTypes = Union[
    str,
    type,
    dt.DataType,
]
