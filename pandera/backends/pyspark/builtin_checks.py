"""Built-in checks for PySpark."""

from typing import Any, TypeVar
from collections.abc import Iterable

import pyspark.sql.types as pst
from pyspark.sql.functions import col

from pandera.api.extensions import register_builtin_check
from pandera.api.pyspark.types import PysparkDataframeColumnObject
from pandera.backends.pyspark.decorators import register_input_datatypes
from pandera.backends.pyspark.utils import convert_to_list

T = TypeVar("T")
ALL_NUMERIC_TYPE = [
    pst.LongType,
    pst.IntegerType,
    pst.ByteType,
    pst.ShortType,
    pst.DoubleType,
    pst.DecimalType,
    pst.FloatType,
]
ALL_DATE_TYPE = [pst.DateType, pst.TimestampType]
BOLEAN_TYPE = pst.BooleanType
BINARY_TYPE = pst.BinaryType
STRING_TYPE = pst.StringType


@register_builtin_check(
    aliases=["eq"],
    error="equal_to({value})",
)
@register_input_datatypes(
    acceptable_datatypes=convert_to_list(
        ALL_NUMERIC_TYPE, ALL_DATE_TYPE, STRING_TYPE, BINARY_TYPE, BOLEAN_TYPE
    )
)
def equal_to(data: PysparkDataframeColumnObject, value: Any) -> bool:
    """Ensure all elements of a column equal a certain value.

    :param data: NamedTuple PysparkDataframeColumnObject contains the dataframe and column name for the check. The key
        to access the dataframe is "dataframe", and the key to access the column name is "column_name".
    :param value: Values in this PySpark data structure must be
        equal to this value.
    """
    cond = col(data.column_name) == value
    return data.dataframe.filter(~cond).limit(1).count() == 0


@register_builtin_check(
    aliases=["ne"],
    error="not_equal_to({value})",
)
@register_input_datatypes(
    acceptable_datatypes=convert_to_list(
        ALL_NUMERIC_TYPE, ALL_DATE_TYPE, STRING_TYPE, BINARY_TYPE, BOLEAN_TYPE
    )
)
def not_equal_to(data: PysparkDataframeColumnObject, value: Any) -> bool:
    """Ensure no element of a column equals a certain value.

    :param data: NamedTuple PysparkDataframeColumnObject contains the dataframe and column name for the check. The key
        to access the dataframe is "dataframe", and the key to access the column name is "column_name".
    :param value: This value must not occur in the checked data structure.
    """
    cond = col(data.column_name) != value
    return data.dataframe.filter(~cond).limit(1).count() == 0


@register_builtin_check(
    aliases=["gt"],
    error="greater_than({min_value})",
)
@register_input_datatypes(
    acceptable_datatypes=convert_to_list(ALL_NUMERIC_TYPE, ALL_DATE_TYPE)
)
def greater_than(data: PysparkDataframeColumnObject, min_value: Any) -> bool:
    """Ensure values of a column are strictly greater than a minimum
    value.

    :param data: NamedTuple PysparkDataframeColumnObject contains the dataframe and column name for the check. The key
        to access the dataframe is "dataframe", and the key to access the column name is "column_name".
    :param min_value: Lower bound to be exceeded. Must be a type comparable
        to the dtype of the :class:`pyspark.sql.Column` to be validated.
    """
    cond = col(data.column_name) > min_value
    return data.dataframe.filter(~cond).limit(1).count() == 0


@register_builtin_check(
    aliases=["ge"],
    error="greater_than_or_equal_to({min_value})",
)
@register_input_datatypes(
    acceptable_datatypes=convert_to_list(ALL_NUMERIC_TYPE, ALL_DATE_TYPE)
)
def greater_than_or_equal_to(
    data: PysparkDataframeColumnObject, min_value: Any
) -> bool:
    """Ensure all values are greater than or equal to a minimum value.

    :param data: NamedTuple PysparkDataframeColumnObject contains the dataframe and column name for the check. The key
        to access the dataframe is "dataframe", and the key to access the column name is "column_name".
    :param min_value: Allowed minimum value. Must be a type comparable
        to the dtype of the :class:`pyspark.sql.Column` to be validated.
    """
    cond = col(data.column_name) >= min_value
    return data.dataframe.filter(~cond).limit(1).count() == 0


@register_builtin_check(
    aliases=["lt"],
    error="less_than({max_value})",
)
@register_input_datatypes(
    acceptable_datatypes=convert_to_list(ALL_NUMERIC_TYPE, ALL_DATE_TYPE)
)
def less_than(data: PysparkDataframeColumnObject, max_value: Any) -> bool:
    """Ensure values of a column are strictly less than a maximum value.

    :param data: NamedTuple PysparkDataframeColumnObject contains the dataframe and column name for the check. The key
        to access the dataframe is "dataframe", and the key to access the column name is "column_name".
    :param max_value: All elements of a column must be strictly smaller
        than this. Must be a type comparable to the dtype of the
        :class:`pyspark.sql.Column` to be validated.
    """
    # test case exists but not detected by pytest so no cover added
    if max_value is None:  # pragma: no cover
        raise ValueError("max_value must not be None")
    cond = col(data.column_name) < max_value
    return data.dataframe.filter(~cond).limit(1).count() == 0


@register_builtin_check(
    aliases=["le"],
    error="less_than_or_equal_to({max_value})",
)
@register_input_datatypes(
    acceptable_datatypes=convert_to_list(ALL_NUMERIC_TYPE, ALL_DATE_TYPE)
)
def less_than_or_equal_to(
    data: PysparkDataframeColumnObject, max_value: Any
) -> bool:
    """Ensure all values are less than or equal to a maximum value.

    :param data: NamedTuple PysparkDataframeColumnObject contains the dataframe and column name for the check. The key
        to access the dataframe is "dataframe", and the key to access the column name is "column_name".
    :param max_value: Upper bound not to be exceeded. Must be a type comparable to the dtype of the
        :class:`pyspark.sql.Column` to be validated.
    """
    # test case exists but not detected by pytest so no cover added
    if max_value is None:  # pragma: no cover
        raise ValueError("max_value must not be None")
    cond = col(data.column_name) <= max_value
    return data.dataframe.filter(~cond).limit(1).count() == 0


@register_builtin_check(
    aliases=["between"],
    error="in_range({min_value}, {max_value})",
)
@register_input_datatypes(
    acceptable_datatypes=convert_to_list(ALL_NUMERIC_TYPE, ALL_DATE_TYPE)
)
def in_range(
    data: PysparkDataframeColumnObject,
    min_value: T,
    max_value: T,
    include_min: bool = True,
    include_max: bool = True,
):
    """Ensure all values of a column are within an interval.

    Both endpoints must be a type comparable to the dtype of the
    :class:`pyspark.sql.Column` to be validated.

    :param data: NamedTuple PysparkDataframeColumnObject contains the dataframe and column name for the check. The key
        to access the dataframe is "dataframe", and the key to access the column name is "column_name".
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
    compare_min = (
        col(data.column_name) >= min_value
        if include_min
        else col(data.column_name) > min_value
    )
    compare_max = (
        col(data.column_name) <= max_value
        if include_max
        else col(data.column_name) < max_value
    )
    return data.dataframe.filter(~(compare_min & compare_max)).limit(1).count() == 0  # type: ignore


@register_builtin_check(
    error="isin({allowed_values})",
)
@register_input_datatypes(
    acceptable_datatypes=convert_to_list(
        ALL_NUMERIC_TYPE, ALL_DATE_TYPE, STRING_TYPE, BINARY_TYPE
    )
)
def isin(data: PysparkDataframeColumnObject, allowed_values: Iterable) -> bool:
    """Ensure only allowed values occur within a series.

    Remember it can be a compute intensive check on large dataset. So, use it with caution.

    This checks whether all elements of a :class:`pyspark.sql.Column`
    are part of the set of elements of allowed values. If allowed
    values is a string, the set of elements consists of all distinct
    characters of the string. Thus only single characters which occur
    in allowed_values at least once can meet this condition. If you
    want to check for substrings use :meth:`Check.str_contains`.

    :param data: NamedTuple PysparkDataframeColumnObject contains the dataframe and column name for the check. The key
        to access the dataframe is "dataframe", and the key to access the column name is "column_name".
    :param allowed_values: The set of allowed values. May be any iterable.
    """
    return (
        data.dataframe.filter(
            ~col(data.column_name).isin(list(allowed_values))
        )
        .limit(1)
        .count()
        == 0
    )


@register_builtin_check(
    error="notin({forbidden_values})",
)
@register_input_datatypes(
    acceptable_datatypes=convert_to_list(
        ALL_NUMERIC_TYPE, ALL_DATE_TYPE, STRING_TYPE, BINARY_TYPE
    )
)
def notin(
    data: PysparkDataframeColumnObject, forbidden_values: Iterable
) -> bool:
    """Ensure some defined values don't occur within a series.

    Remember it can be a compute intensive check on large dataset. So, use it with caution.

    Like :meth:`Check.isin` this check operates on single characters if
    it is applied on strings. If forbidden_values is a string, it is understood
    as set of prohibited characters. Any string of length > 1 can't be in it by
    design.

    :param data: NamedTuple PysparkDataframeColumnObject contains the dataframe and column name for the check. The key
        to access the dataframe is "dataframe", and the key to access the column name is "column_name".
    :param forbidden_values: The set of values which should not occur. May
        be any iterable.
    """
    return (
        data.dataframe.filter(
            col(data.column_name).isin(list(forbidden_values))
        )
        .limit(1)
        .count()
        == 0
    )


@register_builtin_check(
    error="str_contains('{pattern}')",
)
@register_input_datatypes(acceptable_datatypes=convert_to_list(STRING_TYPE))
def str_contains(data: PysparkDataframeColumnObject, pattern: str) -> bool:
    """Ensure that a pattern can be found within each row.

    Remember it can be a compute intensive check on large dataset. So, use it with caution.

    :param data: NamedTuple PysparkDataframeColumnObject contains the dataframe and column name for the check. The key
        to access the dataframe is "dataframe", and the key to access the column name is "column_name".
    :param pattern: Regular expression pattern to use for searching.
    """
    return (
        data.dataframe.filter(~col(data.column_name).rlike(pattern))
        .limit(1)
        .count()
        == 0
    )


@register_builtin_check(
    error="str_startswith('{string}')",
)
@register_input_datatypes(acceptable_datatypes=convert_to_list(STRING_TYPE))
def str_startswith(data: PysparkDataframeColumnObject, string: str) -> bool:
    """Ensure that all values start with a certain string.

    Remember it can be a compute intensive check on large dataset. So, use it with caution.

    :param data: NamedTuple PysparkDataframeColumnObject contains the dataframe and column name for the check. The key
        to access the dataframe is "dataframe", and the key to access the column name is "column_name".
    :param string: String all values should start with.
    """
    cond = col(data.column_name).startswith(string)
    return data.dataframe.filter(~cond).limit(1).count() == 0


@register_builtin_check(error="str_endswith('{string}')")
@register_input_datatypes(acceptable_datatypes=convert_to_list(STRING_TYPE))
def str_endswith(data: PysparkDataframeColumnObject, string: str) -> bool:
    """Ensure that all values end with a certain string.

    Remember it can be a compute intensive check on large dataset. So, use it with caution.

    :param data: NamedTuple PysparkDataframeColumnObject contains the dataframe and column name for the check. The key
        to access the dataframe is "dataframe", and the key to access the column name is "column_name".
    :param string: String all values should end with.
    """
    cond = col(data.column_name).endswith(string)
    return data.dataframe.filter(~cond).limit(1).count() == 0
