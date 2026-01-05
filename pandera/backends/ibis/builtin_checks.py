"""Built-in checks for Ibis."""

import datetime
import re
from collections.abc import Iterable
from typing import Any, Optional, TypeVar, Union

import ibis
from ibis import _
from ibis import selectors as s

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


def _across(table: ibis.Table, selection: str | None, func) -> ibis.Table:
    return table.select(
        s.across(
            s.all() if selection is None else select_column(selection), func
        )
    )


@register_builtin_check(
    aliases=["eq"],
    error="equal_to({value})",
)
def equal_to(data: IbisData, value: Any) -> ibis.Table:
    """Ensure all elements of a column equal a certain value.

    :param data: NamedTuple IbisData contains the table and column name for the check. The key
        to access the table is "table", and the key to access the column name is "key".
    :param value: Values in this Ibis data structure must be
        equal to this value.
    """
    value = _infer_interval_with_mixed_units(value)
    return _across(data.table, data.key, _ == value)


@register_builtin_check(
    aliases=["ne"],
    error="not_equal_to({value})",
)
def not_equal_to(data: IbisData, value: Any) -> ibis.Table:
    """Ensure no element of a column equals a certain value.

    :param data: NamedTuple IbisData contains the table and column name for the check. The key
        to access the table is "table", and the key to access the column name is "key".
    :param value: This value must not occur in the checked data structure.
    """
    value = _infer_interval_with_mixed_units(value)
    return _across(data.table, data.key, _ != value)


@register_builtin_check(
    aliases=["gt"],
    error="greater_than({min_value})",
)
def greater_than(data: IbisData, min_value: Any) -> ibis.Table:
    """Ensure values of a column are strictly greater than a minimum
    value.

    :param data: NamedTuple IbisData contains the table and column name for the check. The key
        to access the table is "table", and the key to access the column name is "key".
    :param min_value: Lower bound to be exceeded. Must be a type comparable
        to the dtype of the :class:`ibis.Column` to be validated.
    """
    value = _infer_interval_with_mixed_units(min_value)
    return _across(data.table, data.key, _ > value)


@register_builtin_check(
    aliases=["ge"],
    error="greater_than_or_equal_to({min_value})",
)
def greater_than_or_equal_to(data: IbisData, min_value: Any) -> ibis.Table:
    """Ensure all values are greater than or equal to a minimum value.

    :param data: NamedTuple IbisData contains the table and column name for the check. The key
        to access the table is "table", and the key to access the column name is "key".
    :param min_value: Allowed minimum value. Must be a type comparable
        to the dtype of the :class:`ibis.Column` to be validated.
    """
    value = _infer_interval_with_mixed_units(min_value)
    return _across(data.table, data.key, _ >= value)


@register_builtin_check(
    aliases=["lt"],
    error="less_than({max_value})",
)
def less_than(data: IbisData, max_value: Any) -> ibis.Table:
    """Ensure values of a column are strictly less than a maximum value.

    :param data: NamedTuple IbisData contains the table and column name for the check. The key
        to access the table is "table", and the key to access the column name is "key".
    :param max_value: All elements of a column must be strictly smaller
        than this. Must be a type comparable to the dtype of the
        :class:`ibis.Column` to be validated.
    """
    value = _infer_interval_with_mixed_units(max_value)
    return _across(data.table, data.key, _ < value)


@register_builtin_check(
    aliases=["le"],
    error="less_than_or_equal_to({max_value})",
)
def less_than_or_equal_to(data: IbisData, max_value: Any) -> ibis.Table:
    """Ensure all values are less than or equal to a maximum value.

    :param data: NamedTuple IbisData contains the table and column name for the check. The key
        to access the table is "table", and the key to access the column name is "key".
    :param max_value: Upper bound not to be exceeded. Must be a type comparable to the dtype of the
        :class:`ibis.Column` to be validated.
    """
    value = _infer_interval_with_mixed_units(max_value)
    return _across(data.table, data.key, _ <= value)


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
) -> ibis.Table:
    """Ensure all values of a column are within an interval.

    Both endpoints must be a type comparable to the dtype of the
    :class:`ibis.Column` to be validated.

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

    return _across(data.table, data.key, func)


@register_builtin_check(
    error="isin({allowed_values})",
)
def isin(data: IbisData, allowed_values: Iterable) -> ibis.Table:
    """Ensure only allowed values occur within a column.

    This checks whether all elements of a :class:`ibis.Column`
    are part of the set of elements of allowed values. If allowed
    values is a string, the set of elements consists of all distinct
    characters of the string. Thus only single characters which occur
    in allowed_values at least once can meet this condition. If you
    want to check for substrings use :meth:`Check.str_contains`.

    :param data: NamedTuple IbisData contains the table and column name for the check. The key
        to access the table is "table", and the key to access the column name is "key".
    :param allowed_values: The set of allowed values. May be any iterable.
    """
    allowed_values = [
        _infer_interval_with_mixed_units(value) for value in allowed_values
    ]
    return _across(data.table, data.key, _.isin(allowed_values))


@register_builtin_check(
    error="notin({allowed_values})",
)
def notin(data: IbisData, forbidden_values: Iterable) -> ibis.Table:
    """Ensure some defined values don't occur within a series.

    Like :meth:`Check.isin`, this check operates on single characters if
    it is applied on strings. If forbidden_values is a string, it is understood
    as set of prohibited characters. Any string of length > 1 can't be in it by
    design.

    :param data: NamedTuple IbisData contains the table and column name for the check. The key
        to access the table is "table", and the key to access the column name is "key".
    :param forbidden_values: The set of values which should not occur. May
        be any iterable.
    """
    forbidden_values = [
        _infer_interval_with_mixed_units(value) for value in forbidden_values
    ]
    return _across(data.table, data.key, _.notin(forbidden_values))


@register_builtin_check(
    error="str_matches({pattern})",
)
def str_matches(
    data: IbisData,
    pattern: Union[str, re.Pattern],
) -> ibis.Table:
    """Ensure all values start with a match of a regular expression pattern.

    :param data: NamedTuple IbisData contains the table and column name for the check. The key
        to access the table is "table", and the key to access the column name is "key".
    :param pattern: Regular expression pattern to use for matching.
    """
    pattern = pattern.pattern if isinstance(pattern, re.Pattern) else pattern
    if not pattern.startswith("^"):
        pattern = f"^{pattern}"
    return _across(data.table, data.key, _.re_search(pattern))


@register_builtin_check(
    error="str_contains({pattern})",
)
def str_contains(
    data: IbisData,
    pattern: Union[str, re.Pattern],
) -> ibis.Table:
    """Ensure that a pattern can be found within each row.

    :param data: NamedTuple IbisData contains the table and column name for the check. The key
        to access the table is "table", and the key to access the column name is "key".
    :param pattern: Regular expression pattern to use for searching.
    """
    pattern = pattern.pattern if isinstance(pattern, re.Pattern) else pattern
    return _across(data.table, data.key, _.re_search(pattern))


@register_builtin_check(
    error="str_startswith({pattern})",
)
def str_startswith(data: IbisData, string: str) -> ibis.Table:
    """Ensure that all values start with a certain string.

    :param data: NamedTuple IbisData contains the table and column name for the check. The key
        to access the table is "table", and the key to access the column name is "key".
    :param pattern: String all values should start with.
    """
    return _across(data.table, data.key, _.startswith(string))


@register_builtin_check(
    error="str_endswith({pattern})",
)
def str_endswith(data: IbisData, string: str) -> ibis.Table:
    """Ensure that all values end with a certain string.

    :param data: NamedTuple IbisData contains the table and column name for the check. The key
        to access the table is "table", and the key to access the column name is "key".
    :param pattern: String all values should end with.
    """
    return _across(data.table, data.key, _.endswith(string))


@register_builtin_check()
def str_length(
    data: IbisData,
    min_value: int | None = None,
    max_value: int | None = None,
    exact_value: int | None = None,
) -> ibis.Table:
    """Ensure that the length of strings is within a specified range.

    :param data: NamedTuple IbisData contains the table and column name for the check. The key
        to access the table is "table", and the key to access the column name is "key".
    :param min_value: Minimum length of strings (inclusive). (default: no minimum)
    :param max_value: Maximum length of strings (inclusive). (default: no maximum)
    :param exact_value: Exact length of strings. (default: no exact value)
    """
    if exact_value is not None:
        func = _.length() == exact_value
        return _across(data.table, data.key, func)

    if min_value is None and max_value is None:
        raise ValueError(
            "Must provide at least one of 'min_value' and 'max_value'"
        )
    elif min_value is None:
        func = _.length() <= max_value
    elif max_value is None:
        func = _.length() >= min_value
    else:
        func = _.length().between(min_value, max_value)

    return _across(data.table, data.key, func)


@register_builtin_check(
    error="unique_values_eq({values})",
)
def unique_values_eq(data: IbisData, values: Iterable) -> ibis.Table:
    """Ensure that unique values in the data object contain all values.

    .. note::
        In contrast with :func:`isin`, this check makes sure that all the items
        in the ``values`` iterable are contained within the series.

    :param data: NamedTuple IbisData contains the table and column name for the check. The key
        to access the table is "table", and the key to access the column name is "key".
    :param values: The set of values that must be present. May be any iterable.
    """
    return (
        set(
            data.table.select(data.key)
            .distinct()[data.key]
            .to_pyarrow()
            .to_pylist()
        )
        == values
    )
