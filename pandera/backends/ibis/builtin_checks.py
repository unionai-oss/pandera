"""Built-in checks for Ibis."""

import datetime
from typing import Any, TypeVar

import ibis
import ibis.expr.types as ir

from pandera.api.extensions import register_builtin_check
from pandera.api.ibis.types import IbisData

T = TypeVar("T")


def _infer_interval_with_mixed_units(value: Any) -> Any:
    """Infer interval with mixed units to prevent Ibis from erroring."""
    if (
        isinstance(value, datetime.timedelta)
        and value.days
        and (value.seconds or value.microseconds)
    ):
        return ibis.interval(value)

    return value


@register_builtin_check(
    aliases=["eq"],
    error="equal_to({value})",
)
def equal_to(data: IbisData, value: Any) -> ir.Table:
    """Ensure all elements of a data container equal a certain value.

    :param data: NamedTuple IbisData contains the table and column name for the check. The key
        to access the table is "table", and the key to access the column name is "key".
    :param value: Values in this Ibis data structure must be
        equal to this value.
    """
    value = _infer_interval_with_mixed_units(value)
    return data.table[data.key] == value


@register_builtin_check(
    aliases=["ne"],
    error="not_equal_to({value})",
)
def not_equal_to(data: IbisData, value: Any) -> ir.Table:
    """Ensure no element of a data container equal a certain value.

    :param data: NamedTuple IbisData contains the table and column name for the check. The key
        to access the table is "table", and the key to access the column name is "key".
    :param value: This value must not occur in the checked data structure.
    """
    value = _infer_interval_with_mixed_units(value)
    return data.table[data.key] != value


@register_builtin_check(
    aliases=["gt"],
    error="greater_than({value})",
)
def greater_than(data: IbisData, min_value: Any) -> ir.Table:
    """Ensure values of a data container are strictly greater than a minimum
    value.

    :param data: NamedTuple IbisData contains the table and column name for the check. The key
        to access the table is "table", and the key to access the column name is "key".
    :param min_value: Lower bound to be exceeded. Must be a type comparable
        to the dtype of the :class:`ir.Column` to be validated.
    """
    value = _infer_interval_with_mixed_units(min_value)
    return data.table[data.key] > value


@register_builtin_check(
    aliases=["ge"],
    error="greater_than_or_equal_to({value})",
)
def greater_than_or_equal_to(data: IbisData, min_value: Any) -> ir.Table:
    """Ensure all values are greater than or equal to a certain value.

    :param data: NamedTuple IbisData contains the table and column name for the check. The key
        to access the table is "table", and the key to access the column name is "key".
    :param min_value: Allowed minimum value. Must be a type comparable
        to the dtype of the :class:`ir.Column` to be validated.
    """
    value = _infer_interval_with_mixed_units(min_value)
    return data.table[data.key] >= value
