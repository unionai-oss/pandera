"""Ibis types."""

import ibis.expr.types as ir
from typing import NamedTuple, Optional, Union
from ibis.expr.datatypes import DataType



class IbisData(NamedTuple):
    table: ir.Table
    key: Optional[str] = None

class CheckResult(NamedTuple):
    """Check result for user-defined checks."""

    check_output: ir.Table
    check_passed: ir.Table
    checked_object: ir.Table
    failure_cases: ir.Table

IbisCheckObjects = Union[ir.Table]

IbisDtypeInputTypes = Union[
    str,
    type,
    ir.Table,
    ir.Column,
    ir.Literal,
    DataType
]