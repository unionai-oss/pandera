"""Built-in checks for narwhals backend."""

from typing import Any, Dict, List, Optional, Union

import narwhals as nw

from pandera.api.narwhals.types import NarwhalsData
from pandera.api.extensions import register_builtin_check


def _element_check(
    check_fn,
    narwhals_data: NarwhalsData,
    **kwargs,
) -> nw.DataFrame[Any]:
    """Apply element-wise check to narwhals data."""
    # Placeholder implementation
    return narwhals_data.dataframe


def _aggregate_check(
    check_fn,
    narwhals_data: NarwhalsData,
    **kwargs,
) -> nw.DataFrame[Any]:
    """Apply aggregate check to narwhals data."""
    # Placeholder implementation
    return narwhals_data.dataframe


@register_builtin_check(error="equal_to({value})")
def equal_to(
    narwhals_data: NarwhalsData, value: Any, **kwargs
) -> nw.DataFrame[Any]:
    """Check if values are equal to a specified value."""
    # Placeholder implementation
    return narwhals_data.dataframe


@register_builtin_check(error="not_equal_to({value})")
def not_equal_to(
    narwhals_data: NarwhalsData, value: Any, **kwargs
) -> nw.DataFrame[Any]:
    """Check if values are not equal to a specified value."""
    # Placeholder implementation
    return narwhals_data.dataframe


@register_builtin_check(error="greater_than({value})")
def greater_than(
    narwhals_data: NarwhalsData, value: Any, **kwargs
) -> nw.DataFrame[Any]:
    """Check if values are greater than a specified value."""
    # Placeholder implementation
    return narwhals_data.dataframe


@register_builtin_check(error="greater_than_or_equal_to({value})")
def greater_than_or_equal_to(
    narwhals_data: NarwhalsData, value: Any, **kwargs
) -> nw.DataFrame[Any]:
    """Check if values are greater than or equal to a specified value."""
    # Placeholder implementation
    return narwhals_data.dataframe


@register_builtin_check(error="less_than({value})")
def less_than(
    narwhals_data: NarwhalsData, value: Any, **kwargs
) -> nw.DataFrame[Any]:
    """Check if values are less than a specified value."""
    # Placeholder implementation
    return narwhals_data.dataframe


@register_builtin_check(error="less_than_or_equal_to({value})")
def less_than_or_equal_to(
    narwhals_data: NarwhalsData, value: Any, **kwargs
) -> nw.DataFrame[Any]:
    """Check if values are less than or equal to a specified value."""
    # Placeholder implementation
    return narwhals_data.dataframe


@register_builtin_check(error="in_range({min_val}, {max_val})")
def in_range(
    narwhals_data: NarwhalsData,
    min_val: Any,
    max_val: Any,
    include_min: bool = True,
    include_max: bool = True,
    **kwargs,
) -> nw.DataFrame[Any]:
    """Check if values are within a specified range."""
    # Placeholder implementation
    return narwhals_data.dataframe


@register_builtin_check(error="isin({allowed_values})")
def isin(
    narwhals_data: NarwhalsData, allowed_values: List[Any], **kwargs
) -> nw.DataFrame[Any]:
    """Check if values are in a list of allowed values."""
    # Placeholder implementation
    return narwhals_data.dataframe


@register_builtin_check(error="notin({forbidden_values})")
def notin(
    narwhals_data: NarwhalsData, forbidden_values: List[Any], **kwargs
) -> nw.DataFrame[Any]:
    """Check if values are not in a list of forbidden values."""
    # Placeholder implementation
    return narwhals_data.dataframe


@register_builtin_check(error="str_contains({pattern})")
def str_contains(
    narwhals_data: NarwhalsData, pattern: str, case: bool = True, **kwargs
) -> nw.DataFrame[Any]:
    """Check if string values contain a pattern."""
    # Placeholder implementation
    return narwhals_data.dataframe


@register_builtin_check(error="str_matches({pattern})")
def str_matches(
    narwhals_data: NarwhalsData, pattern: str, case: bool = True, **kwargs
) -> nw.DataFrame[Any]:
    """Check if string values match a pattern."""
    # Placeholder implementation
    return narwhals_data.dataframe


@register_builtin_check(error="str_length({min_val}, {max_val})")
def str_length(
    narwhals_data: NarwhalsData,
    min_val: Optional[int] = None,
    max_val: Optional[int] = None,
    **kwargs,
) -> nw.DataFrame[Any]:
    """Check if string values have a specified length."""
    # Placeholder implementation
    return narwhals_data.dataframe
