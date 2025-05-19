"""Built-in checks for Ibis."""

import datetime
from typing import Any, Iterable, Optional, TypeVar

import ibis
import ibis.expr.types as ir
from ibis import _, selectors as s
from ibis.common.selectors import Selector

from pandera.api.extensions import register_builtin_check
from pandera.api.ibis.types import IbisData
from pandera.backends.ibis.utils import select_column

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
    return data.table.select(s.across(_selector(data.key), _ == value))


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
    return data.table.select(s.across(_selector(data.key), _ != value))


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
    return data.table.select(s.across(_selector(data.key), _ > value))


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
    return data.table.select(s.across(_selector(data.key), _ >= value))


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
    return data.table.select(s.across(_selector(data.key), _ < value))


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
    return data.table.select(s.across(_selector(data.key), _ <= value))


@register_builtin_check(
    aliases=["between"],
    error="in_range({min_value}, {max_value})",
)
def in_range(
    data: IbisData,
    min_value: T,
    max_value: T,
    include_min: bool = True,
    include_max: bool = True,
) -> ir.Table:
    """Ensure all values of a column are within an interval.

    Both endpoints must be a type comparable to the dtype of the
    :class:`ir.Column` to be validated.

    :param data: NamedTuple IbisData contains the table and column name for the check. The key
        to access the table is "table", and the key to access the column name is "key".
    :param min_value: Left / lower endpoint of the interval.
    :param max_value: Right / upper endpoint of the interval. Must not be
        smaller than min_value.
    :param include_min: Defines whether min_value is also an allowed value
        (the default) or whether all values must be strictly greater than
        min_value.
    :param include_max: Defines whether min_value is also an allowed value
        (the default) or whether all values must be strictly smaller than
        max_value.
    """
    min_value = _infer_interval_with_mixed_units(min_value)
    max_value = _infer_interval_with_mixed_units(max_value)
    if include_min and include_max:
        func = _.between(min_value, max_value)
    else:
        compare_min = _ >= min_value if include_min else _ > min_value
        compare_max = _ <= max_value if include_max else _ < max_value
        func = compare_min & compare_max
    return data.table.select(s.across(_selector(data.key), func))


@register_builtin_check(
    error="isin({allowed_values})",
)
def isin(data: IbisData, allowed_values: Iterable) -> ir.Table:
    """Ensure only allowed values occur within a column.

    This checks whether all elements of a :class:`ir.Column`
    are part of the set of elements of allowed values. If allowed
    values is a string, the set of elements consists of all distinct
    characters of the string. Thus only single characters which occur
    in allowed_values at least once can meet this condition. If you
    want to check for substrings use :meth:`Check.str_contains`.

    :param data: NamedTuple IbisData contains the table and column name for the check. The key
        to access the table is "table", and the key to access the column name is "key".
    :param allowed_values: The set of allowed values. May be any iterable.
    """
    allowed_values = [_infer_interval_with_mixed_units(value) for value in allowed_values]
    return data.table.select(s.across(_selector(data.key), _.isin(allowed_values)))
