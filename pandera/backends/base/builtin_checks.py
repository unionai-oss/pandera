"""Built-in check functions base implementation.

This module contains check function abstract definitions that correspond to
the pandera.api.base.checks.Check methods. These functions do not actually
implement any validation logic and serve as the entrypoint for dispatching
specific implementations based on the data object type, e.g.
`pandas.DataFrame`s.
"""

import re
from collections.abc import Iterable
from typing import Any, TypeVar, Union

import narwhals as nw

from pandera.api.base.types import CheckData
from pandera.api.checks import Check

T = TypeVar("T")


@Check.register_builtin_check_fn
def equal_to(data: CheckData, value: Any) -> Any:
    """Ensure all elements of a column equal a certain value."""
    nw_frame = nw.from_native(data.frame)
    result = nw_frame.select(nw.col(data.key) == value)
    return nw.to_native(result)


@Check.register_builtin_check_fn
def not_equal_to(data: CheckData, value: Any) -> Any:
    """Ensure no element of a column equals a certain value."""
    nw_frame = nw.from_native(data.frame)
    result = nw_frame.select(nw.col(data.key) != value)
    return nw.to_native(result)


@Check.register_builtin_check_fn
def greater_than(data: CheckData, min_value: Any) -> Any:
    """Ensure values are strictly greater than a minimum value."""
    nw_frame = nw.from_native(data.frame)
    result = nw_frame.select(nw.col(data.key) > min_value)
    return nw.to_native(result)


@Check.register_builtin_check_fn
def greater_than_or_equal_to(data: CheckData, min_value: Any) -> Any:
    """Ensure all values are greater than or equal to a minimum value."""
    nw_frame = nw.from_native(data.frame)
    result = nw_frame.select(nw.col(data.key) >= min_value)
    return nw.to_native(result)


@Check.register_builtin_check_fn
def less_than(data: CheckData, max_value: Any) -> Any:
    """Ensure values are strictly less than a maximum value."""
    nw_frame = nw.from_native(data.frame)
    result = nw_frame.select(nw.col(data.key) < max_value)
    return nw.to_native(result)


@Check.register_builtin_check_fn
def less_than_or_equal_to(data: CheckData, max_value: Any) -> Any:
    """Ensure all values are less than or equal to a maximum value."""
    nw_frame = nw.from_native(data.frame)
    result = nw_frame.select(nw.col(data.key) <= max_value)
    return nw.to_native(result)


@Check.register_builtin_check_fn
def in_range(
    data: Any,
    min_value: T,
    max_value: T,
    include_min: bool = True,
    include_max: bool = True,
) -> Any:
    raise NotImplementedError


@Check.register_builtin_check_fn
def isin(data: Any, allowed_values: Iterable) -> Any:
    raise NotImplementedError


@Check.register_builtin_check_fn
def notin(data: Any, forbidden_values: Iterable) -> Any:
    raise NotImplementedError


@Check.register_builtin_check_fn
def str_matches(data: Any, pattern: Union[str, re.Pattern]) -> Any:
    raise NotImplementedError


@Check.register_builtin_check_fn
def str_contains(data: Any, pattern: Union[str, re.Pattern]) -> Any:
    raise NotImplementedError


@Check.register_builtin_check_fn
def str_startswith(data: Any, string: str) -> Any:
    raise NotImplementedError


@Check.register_builtin_check_fn
def str_endswith(data: Any, string: str) -> Any:
    raise NotImplementedError


@Check.register_builtin_check_fn
def str_length(
    data: Any,
    min_value: int | None = None,
    max_value: int | None = None,
    exact_value: int | None = None,
) -> Any:
    raise NotImplementedError


@Check.register_builtin_check_fn
def unique_values_eq(data: Any, values: Iterable) -> Any:
    raise NotImplementedError
