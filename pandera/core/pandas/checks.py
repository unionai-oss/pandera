"""Pandas implementation of built-in checks"""

import operator
import re
from typing import Any, Iterable, TypeVar, Union

import pandas as pd

import pandera.strategies as st
import pandera.typing
from pandera.core.checks import register_check

PandasData = Union[pd.Series, pd.DataFrame]


if pandera.typing.modin.MODIN_INSTALLED:
    import modin.pandas as mpd

    PandasData = Union[PandasData, mpd.Series, mpd.DataFrame]


if pandera.typing.pyspark.PYSPARK_INSTALLED:
    import pyspark.pandas as ppd

    PandasData = Union[PandasData, ppd.Series, ppd.DataFrame]


T = TypeVar("T")


@register_check(
    aliases=["eq"],
    strategy=st.eq_strategy,
    error="equal_to({value})",
)
def equal_to(data: PandasData, value: Any) -> PandasData:
    """Ensure all elements of a data container equal a certain value.

    *New in version 0.4.5*

    Alias: ``equal_to``

    :param value: values in this pandas data structure must be
        equal to this value.
    """
    return data == value


@register_check(
    aliases=["ne"],
    strategy=st.ne_strategy,
    error="not_equal_to({value})",
)
def not_equal_to(data: PandasData, value: Any) -> PandasData:
    """Ensure no elements of a data container equals a certain value.

    *New in version 0.4.5*

    Alias: ``not_equal_to``

    :param value: This value must not occur in the checked
        :class:`pandas.Series`.
    """
    return data != value


def gt_ge_pre_init_hook(statistics_kwargs):
    """Pre-init hook for greater than/greater or equal to check."""
    if statistics_kwargs["min_value"] is None:
        raise ValueError("min_value must not be None")
    return statistics_kwargs


@register_check(
    pre_init_hook=gt_ge_pre_init_hook,
    aliases=["gt"],
    strategy=st.gt_strategy,
    error="greater_than({min_value})",
)
def greater_than(data: PandasData, min_value: Any) -> PandasData:
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
    pre_init_hook=gt_ge_pre_init_hook,
    aliases=["ge"],
    strategy=st.ge_strategy,
    error="greater_than_or_equal_to({min_value})",
)
def greater_than_or_equal_to(data: PandasData, min_value: Any) -> PandasData:
    """Ensure all values are greater or equal a certain value.

    *New in version 0.4.5*

    Alias: ``greater_than_or_equal_to``

    :param min_value: Allowed minimum value for values of a series. Must be
        a type comparable to the dtype of the :class:`pandas.Series` to be
        validated.
    """
    return data >= min_value


def lt_le_pre_init_hook(statistics_kwargs):
    """Pre-init hook for less than/less than or equal to check."""
    if statistics_kwargs["max_value"] is None:
        raise ValueError("max_value must not be None")
    return statistics_kwargs


@register_check(
    pre_init_hook=lt_le_pre_init_hook,
    aliases=["lt"],
    strategy=st.lt_strategy,
    error="less_than({max_value})",
)
def less_than(data: PandasData, max_value: Any) -> PandasData:
    """Ensure values of a series are strictly below a maximum value.

    *New in version 0.4.5*

    Alias: ``lt``

    :param max_value: All elements of a series must be strictly smaller
        than this. Must be a type comparable to the dtype of the
        :class:`pandas.Series` to be validated.
    """
    if max_value is None:
        raise ValueError("max_value must not be None")
    return data < max_value


@register_check(
    pre_init_hook=lt_le_pre_init_hook,
    aliases=["le"],
    strategy=st.le_strategy,
    error="less_than_or_equal_to({max_value})",
)
def less_than_or_equal_to(data: PandasData, max_value: Any) -> PandasData:
    """Ensure values of a series are strictly below a maximum value.

    *New in version 0.4.5*

    Alias: ``lt``

    :param max_value: Upper bound not to be exceeded. Must be a type
        comparable to the dtype of the :class:`pandas.Series` to be
        validated.
    """
    if max_value is None:
        raise ValueError("max_value must not be None")
    return data <= max_value


def in_range_pre_init_hook(statistics_kwargs):
    """Pre-init hook for ``in_range`` check."""
    min_value = statistics_kwargs["min_value"]
    max_value = statistics_kwargs["max_value"]
    include_min = statistics_kwargs["include_min"]
    include_max = statistics_kwargs["include_max"]

    if min_value is None:
        raise ValueError("min_value must not be None")
    if max_value is None:
        raise ValueError("max_value must not be None")
    if max_value < min_value or (  # type: ignore
        min_value == max_value and (not include_min or not include_max)
    ):
        raise ValueError(
            f"The combination of min_value = {min_value} and "
            f"max_value = {max_value} defines an empty interval!"
        )
    return statistics_kwargs


@register_check(
    pre_init_hook=in_range_pre_init_hook,
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
    # Using functions from operator module to keep conditions out of the
    # closure
    left_op = operator.le if include_min else operator.lt
    right_op = operator.ge if include_max else operator.gt
    return left_op(min_value, data) & right_op(max_value, data)  # type: ignore


def isin_pre_init_hook(statistics_kwargs):
    """Pre-init hook for ``isin`` check."""
    allowed_values = statistics_kwargs["allowed_values"]
    try:
        allowed_values = frozenset(allowed_values)
    except TypeError as exc:
        raise ValueError(
            f"Argument allowed_values must be iterable. Got {allowed_values}"
        ) from exc
    return {"allowed_values": allowed_values}


@register_check(
    pre_init_hook=isin_pre_init_hook,
    strategy=st.isin_strategy,
    error="isin({allowed_values})",
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
    return data.isin(allowed_values)


def notin_pre_init_hook(statistics_kwargs):
    """Pre-init hook for ``notin`` check."""
    forbidden_values = statistics_kwargs["forbidden_values"]
    try:
        forbidden_values = frozenset(forbidden_values)
    except TypeError as exc:
        raise ValueError(
            f"Argument forbidden_values must be iterable. Got {forbidden_values}"
        ) from exc
    return {"forbidden_values": forbidden_values}


@register_check(
    pre_init_hook=notin_pre_init_hook,
    strategy=st.notin_strategy,
    error="notin({forbidden_values})",
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
    return ~data.isin(forbidden_values)


def str_regex_pre_init_hook(statistics_kwargs):
    """Pre-init hook for string regex checks."""
    pattern = statistics_kwargs["pattern"]
    try:
        regex = re.compile(pattern)
    except TypeError as exc:
        raise ValueError(
            f'pattern="{pattern}" cannot be compiled as regular expression'
        ) from exc
    return {"pattern": regex}


@register_check(
    pre_init_hook=str_regex_pre_init_hook,
    strategy=st.str_matches_strategy,
    error="str_matches('{pattern}')",
)
def str_matches(
    data: PandasData,
    pattern: Union[str, re.Pattern],
) -> PandasData:
    """Ensure that string values match a regular expression.

    :param pattern: Regular expression pattern to use for matching
    :param kwargs: key-word arguments passed into the `Check` initializer.
    """
    return data.str.match(pattern, na=False)


@register_check(
    pre_init_hook=str_regex_pre_init_hook,
    strategy=st.str_contains_strategy,
    error="str_contains('{pattern}')",
)
def str_contains(
    data: PandasData,
    pattern: Union[str, re.Pattern],
) -> PandasData:
    """Ensure that a pattern can be found within each row.

    :param pattern: Regular expression pattern to use for searching
    :param kwargs: key-word arguments passed into the `Check` initializer.
    """
    return data.str.contains(pattern, na=False)


@register_check(
    strategy=st.str_startswith_strategy,
    error="str_startswith('{string}')",
)
def str_startswith(data: PandasData, string: str) -> PandasData:
    """Ensure that all values start with a certain string.

    :param string: String all values should start with
    :param kwargs: key-word arguments passed into the `Check` initializer.
    """
    return data.str.startswith(string, na=False)


@register_check(
    strategy=st.str_endswith_strategy, error="str_endswith('{string}')"
)
def str_endswith(data: PandasData, string: str) -> PandasData:
    """Ensure that all values end with a certain string.

    :param string: String all values should end with
    :param kwargs: key-word arguments passed into the `Check` initializer.
    """
    return data.str.endswith(string, na=False)


def str_length_pre_init_hook(statistics_kwargs):
    """Pre-init hook for ``str_length`` check."""
    min_value = statistics_kwargs["min_value"]
    max_value = statistics_kwargs["max_value"]
    if min_value is None and max_value is None:
        raise ValueError(
            "At least a minimum or a maximum need to be specified. Got "
            "None."
        )
    return statistics_kwargs


@register_check(
    pre_init_hook=str_length_pre_init_hook,
    strategy=st.str_length_strategy,
    error="str_length({min_value}, {max_value})",
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


def unique_values_eq_init_hook(statistics_kwargs):
    """Pre-init hook for ``unique_values`` check."""
    values = statistics_kwargs["values"]
    try:
        values = frozenset(values)
    except TypeError as exc:
        raise ValueError(
            f"Argument values must be iterable. Got {values}"
        ) from exc
    return {"values": values}


@register_check(
    pre_init_hook=unique_values_eq_init_hook,
    error="unique_values_eq({values})",
)
def unique_values_eq(data: PandasData, values: Iterable):
    """Ensure that unique values in the data object contain all values.

    .. note::
        In constrast with :func:`isin`, this check makes sure that all the items
        in the ``values`` iterable are contained within the series.

    :param values: The set of values that must be present. Maybe any iterable.
    """
    return set(data.unique()) == values  # type: ignore[return-value]
