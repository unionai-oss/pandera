"""Built-in checks for Ibis."""

import datetime
from typing import Any, Optional, TypeVar

import ibis
import ibis.expr.types as ir
from ibis import _, selectors as s
from ibis.common.selectors import Selector

from pandera.api.extensions import register_builtin_check
from pandera.api.ibis.types import IbisData
from pandera.backends.ibis.utils import select_column
from pandera.constants import check_col_name


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


def _selector(key: Optional[str]) -> Selector:
    return s.all() if key is None else select_column(key)


@register_builtin_check(
    aliases=["eq"],
    error="equal_to({value})",
)
def equal_to(data: IbisData, value: Any) -> ir.Table:
    """Ensure all elements of a column equal a certain value.

    :param data: NamedTuple IbisData contains the table and column name for the check. The key
        to access the table is "table", and the key to access the column name is "key".
    :param value: Values in this Ibis data structure must be
        equal to this value.
    """
    value = _infer_interval_with_mixed_units(value)
    return data.table.mutate(
        s.across(_selector(data.key), _ == value, names=check_col_name)
    )


@register_builtin_check(
    aliases=["ne"],
    error="not_equal_to({value})",
)
def not_equal_to(data: IbisData, value: Any) -> ir.Table:
    """Ensure no element of a column equals a certain value.

    :param data: NamedTuple IbisData contains the table and column name for the check. The key
        to access the table is "table", and the key to access the column name is "key".
    :param value: This value must not occur in the checked data structure.
    """
    value = _infer_interval_with_mixed_units(value)
    return data.table.mutate(
        s.across(_selector(data.key), _ != value, names=check_col_name)
    )


@register_builtin_check(
    aliases=["gt"],
    error="greater_than({min_value})",
)
def greater_than(data: IbisData, min_value: Any) -> ir.Table:
    """Ensure values of a column are strictly greater than a minimum
    value.

    :param data: NamedTuple IbisData contains the table and column name for the check. The key
        to access the table is "table", and the key to access the column name is "key".
    :param min_value: Lower bound to be exceeded. Must be a type comparable
        to the dtype of the :class:`ir.Column` to be validated.
    """
    value = _infer_interval_with_mixed_units(min_value)
    return data.table.mutate(
        s.across(_selector(data.key), _ > value, names=check_col_name)
    )


@register_builtin_check(
    aliases=["ge"],
    error="greater_than_or_equal_to({min_value})",
)
def greater_than_or_equal_to(data: IbisData, min_value: Any) -> ir.Table:
    """Ensure all values are greater than or equal to a minimum value.

    :param data: NamedTuple IbisData contains the table and column name for the check. The key
        to access the table is "table", and the key to access the column name is "key".
    :param min_value: Allowed minimum value. Must be a type comparable
        to the dtype of the :class:`ir.Column` to be validated.
    """
    value = _infer_interval_with_mixed_units(min_value)
    return data.table.mutate(
        s.across(_selector(data.key), _ >= value, names=check_col_name)
    )


@register_builtin_check(
    aliases=["lt"],
    error="less_than({max_value})",
)
def less_than(data: IbisData, max_value: Any) -> ir.Table:
    """Ensure values of a column are strictly less than a maximum value.

    :param data: NamedTuple IbisData contains the table and column name for the check. The key
        to access the table is "table", and the key to access the column name is "key".
    :param max_value: All elements of a column must be strictly smaller
        than this. Must be a type comparable to the dtype of the
        :class:`ir.Column` to be validated.
    """
    value = _infer_interval_with_mixed_units(max_value)
    return data.table.mutate(
        s.across(_selector(data.key), _ < value, names=check_col_name)
    )


@register_builtin_check(
    aliases=["le"],
    error="less_than_or_equal_to({max_value})",
)
def less_than_or_equal_to(data: IbisData, max_value: Any) -> ir.Table:
    """Ensure all values are less than or equal to a maximum value.

    :param data: NamedTuple IbisData contains the table and column name for the check. The key
        to access the table is "table", and the key to access the column name is "key".
    :param max_value: Upper bound not to be exceeded. Must be a type comparable to the dtype of the
        :class:`ir.Column` to be validated.
    """
    value = _infer_interval_with_mixed_units(max_value)
    return data.table.mutate(
        s.across(_selector(data.key), _ <= value, names=check_col_name)
    )
