"""Pandas implementation of built-in checks"""

import operator
import re
from inspect import signature, Parameter, Signature
from functools import partial, wraps
from typing import Any, Callable, Dict, Iterable, List, Optional, Type, TypeVar, Union

import pandas as pd

import pandera.strategies as st
from pandera.core.checks import register_check


PandasData = Union[pd.Series, pd.DataFrame]


T = TypeVar("T")


@register_check(
    aliases=["equal_to"],
    strategy=st.eq_strategy,
    error="equal_to({value})",
)
def eq(data: PandasData, value: Any) -> PandasData:
    """Ensure all elements of a data container equal a certain value.

    *New in version 0.4.5*

    Alias: ``equal_to``
    
    :param value: values in this pandas data structure must be
        equal to this value.
    """
    return data == value


@register_check(
    aliases=["not_equal_to"],
    strategy=st.ne_strategy,
    error="not_equal_to({value})",
)
def ne(data: PandasData, value: Any) -> PandasData:
    """Ensure no elements of a data container equals a certain value.

    *New in version 0.4.5*

    Alias: ``not_equal_to``

    :param value: This value must not occur in the checked
        :class:`pandas.Series`.
    """
    return data != value


@register_check(
    aliases=["greater_than"],
    strategy=st.gt_strategy,
    error="greater_than({min_value})",
)
def gt(data: PandasData, min_value: Any) -> PandasData:
    """
    Ensure values of a data container are strictly greater than a minimum
    value.

    *New in version 0.4.5*

    Alias: ``greater_than``

    :param min_value: Lower bound to be exceeded. Must be a type comparable
        to the dtype of the :class:`pandas.Series` to be validated (e.g. a
        numerical type for float or int and a datetime for datetime).
    """
    return data > min_value


@register_check(
    aliases=["greater_than_or_equal_to"],
    strategy=st.ge_strategy,
    error="greater_than_or_equal_to({min_value})",
)
def ge(data: PandasData, min_value: Any) -> PandasData:
    """Ensure all values are greater or equal a certain value.

    *New in version 0.4.5*

    Alias: ``greater_than_or_equal_to``

    :param min_value: Allowed minimum value for values of a series. Must be
        a type comparable to the dtype of the :class:`pandas.Series` to be
        validated.
    """
    return data >= min_value


@register_check(
    aliases=["less_than"],
    strategy=st.lt_strategy,
    error="less_than({max_value})",
)
def lt(data: PandasData, max_value: Any) -> PandasData:
    """Ensure values of a series are strictly below a maximum value.

    *New in version 0.4.5*

    Alias: ``lt``

    :param max_value: All elements of a series must be strictly smaller
        than this. Must be a type comparable to the dtype of the
        :class:`pandas.Series` to be validated.
    """
    return data < max_value


@register_check(
    aliases=["less_than_or_equal_to"],
    strategy=st.le_strategy,
    error="less_than_or_equal_to({max_value})",
)
def le(data: PandasData, max_value: Any) -> PandasData:
    """Ensure values of a series are strictly below a maximum value.

    *New in version 0.4.5*

    Alias: ``lt``

    :param max_value: Upper bound not to be exceeded. Must be a type
        comparable to the dtype of the :class:`pandas.Series` to be
        validated.
    """
    return data <= max_value


@register_check(
    aliases=["between"],
    strategy=st.in_range_strategy,
    error="in_range({min_value}, {max_value})",
)
def in_range(
    data: PandasData,
    min_value: T,
    max_value: T,
    include_min: bool = True,
    include_max: bool = True,
):
    """Ensure all values of a series are within an interval.

    Both endpoints must be a type comparable to the dtype of the
    :class:`pandas.Series` to be validated.

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
    if min_value is None:
        raise ValueError("min_value must not be None")
    if max_value is None:
        raise ValueError("max_value must not be None")
    if max_value < min_value or (
        min_value == max_value and (not include_min or not include_max)
    ):
        raise ValueError(
            f"The combination of min_value = {min_value} and "
            f"max_value = {max_value} defines an empty interval!"
        )

    # Using functions from operator module to keep conditions out of the
    # closure
    left_op = operator.le if include_min else operator.lt
    right_op = operator.ge if include_max else operator.gt
    return left_op(min_value, data) & right_op(max_value, data)


@register_check(
    strategy=st.isin_strategy,
    error="isin({allowed_values})"
)
def isin(data: PandasData, allowed_values: Iterable) -> PandasData:
    """Ensure only allowed values occur within a series.

    This checks whether all elements of a :class:`pandas.Series`
    are part of the set of elements of allowed values. If allowed
    values is a string, the set of elements consists of all distinct
    characters of the string. Thus only single characters which occur
    in allowed_values at least once can meet this condition. If you
    want to check for substrings use :meth:`Check.str_contains`.

    :param allowed_values: The set of allowed values. May be any iterable.
    :param kwargs: key-word arguments passed into the `Check` initializer.
    """
    try:
        allowed_values = frozenset(allowed_values)
    except TypeError as exc:
        raise ValueError(
            f"Argument allowed_values must be iterable. Got {allowed_values}"
        ) from exc
    return data.isin(allowed_values)


@register_check(
    strategy=st.notin_strategy,
    error="notin({forbidden_values})"
)
def notin(data: PandasData, forbidden_values: Iterable) -> PandasData:
    """Ensure some defined values don't occur within a series.

    Like :meth:`Check.isin` this check operates on single characters if
    it is applied on strings. If forbidden_values is a string, it is understood
    as set of prohibited characters. Any string of length > 1 can't be in it by
    design.

    :param forbidden_values: The set of values which should not occur. May
        be any iterable.
    :param raise_warning: if True, check raises UserWarning instead of
        SchemaError on validation.
    """
    try:
        forbidden_values = frozenset(forbidden_values)
    except TypeError as exc:
        raise ValueError(
            f"Argument forbidden_values must be iterable. Got {forbidden_values}"
        ) from exc
    return ~data.isin(forbidden_values)


@register_check(
    strategy=st.str_matches_strategy,
    error="str_matches({regex})"
)
def str_matches(data: PandasData, pattern: str) -> PandasData:
    """Ensure that string values match a regular expression.

    :param pattern: Regular expression pattern to use for matching
    :param kwargs: key-word arguments passed into the `Check` initializer.
    """
    try:
        regex = re.compile(pattern)
    except TypeError as exc:
        raise ValueError(
            f'pattern="{pattern}" cannot be compiled as regular expression'
        ) from exc
    return data.str.match(regex, na=False)


@register_check(
    strategy=st.str_contains_strategy,
    error="str_contains({regex})"
)
def str_contains(data: PandasData, pattern: str) -> PandasData:
    """Ensure that a pattern can be found within each row.

    :param pattern: Regular expression pattern to use for searching
    :param kwargs: key-word arguments passed into the `Check` initializer.
    """
    try:
        regex = re.compile(pattern)
    except TypeError as exc:
        raise ValueError(
            f'pattern="{pattern}" cannot be compiled as regular expression'
        ) from exc
    return data.str.contains(regex, na=False)


@register_check(
    strategy=st.str_startswith_strategy,
    error="str_startswith({string})"
)
def str_startswith(data: PandasData, string: str) -> PandasData:
    """Ensure that all values start with a certain string.

    :param string: String all values should start with
    :param kwargs: key-word arguments passed into the `Check` initializer.
    """
    return data.str.startswith(string, na=False)


@register_check(
    strategy=st.str_endswith_strategy,
    error="str_endswith({string})"
)
def str_endswith(data: PandasData, string: str) -> PandasData:
    """Ensure that all values end with a certain string.

    :param string: String all values should end with
    :param kwargs: key-word arguments passed into the `Check` initializer.
    """
    return data.str.endswith(string, na=False)


@register_check(
    strategy=st.str_length_strategy,
    error="str_length({string})"
)
def str_length(
    data: PandasData,
    min_value: int = None,
    max_value: int = None,
) -> PandasData:
    """Ensure that the length of strings is within a specified range.

    :param min_value: Minimum length of strings (default: no minimum)
    :param max_value: Maximum length of strings (default: no maximum)
    """
    if min_value is None and max_value is None:
        raise ValueError(
            "At least a minimum or a maximum need to be specified. Got "
            "None."
        )
    str_len = data.str.len()
    if max_value is None:
        return str_len >= min_value
    elif min_value is None:
        return str_len <= max_value
    return (str_len <= max_value) & (str_len >= min_value)
