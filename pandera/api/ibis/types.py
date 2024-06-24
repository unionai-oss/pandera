"""Ibis types."""

from typing import NamedTuple, Union

import ibis.expr.datatypes as dt
import ibis.expr.types as ir


class CheckResult(NamedTuple):
    """Check result for user-defined checks."""

    check_output: ir.Table
    check_passed: ir.Table
    checked_object: ir.Table
    failure_cases: ir.Table


IbisDtypeInputTypes = Union[
    str,
    type,
    dt.DataType,
]
