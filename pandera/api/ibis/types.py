"""Ibis types."""

from typing import NamedTuple

import ibis.expr.types as ir


class CheckResult(NamedTuple):
    """Check result for user-defined checks."""

    check_output: ir.Table
    check_passed: ir.Table
    checked_object: ir.Table
    failure_cases: ir.Table
