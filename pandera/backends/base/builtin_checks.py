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

from pandera.api.checks import Check

T = TypeVar("T")


@Check.register_builtin_check_fn
def equal_to(data: Any, value: Any) -> Any:
    raise NotImplementedError


@Check.register_builtin_check_fn
def not_equal_to(data: Any, value: Any) -> Any:
    raise NotImplementedError


@Check.register_builtin_check_fn
def greater_than(data: Any, min_value: Any) -> Any:
    raise NotImplementedError


@Check.register_builtin_check_fn
def greater_than_or_equal_to(data: Any, min_value: Any) -> Any:
    raise NotImplementedError


@Check.register_builtin_check_fn
def less_than(data: Any, max_value: Any) -> Any:
    raise NotImplementedError


@Check.register_builtin_check_fn
def less_than_or_equal_to(data: Any, max_value: Any) -> Any:
    raise NotImplementedError


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


@Check.register_builtin_check_fn
def has_dims(data: Any, dims: tuple[str, ...]) -> Any:
    raise NotImplementedError


@Check.register_builtin_check_fn
def has_coords(data: Any, coords: tuple[str, ...]) -> Any:
    raise NotImplementedError


@Check.register_builtin_check_fn
def has_attrs(data: Any, attrs: dict[str, Any]) -> Any:
    raise NotImplementedError


@Check.register_builtin_check_fn
def ndim(data: Any, n: int) -> Any:
    raise NotImplementedError


@Check.register_builtin_check_fn
def dim_size(data: Any, dim: str, size: int) -> Any:
    raise NotImplementedError


@Check.register_builtin_check_fn
def is_monotonic(
    data: Any,
    dim: str,
    increasing: bool = True,
) -> Any:
    raise NotImplementedError


@Check.register_builtin_check_fn
def no_duplicates_in_coord(data: Any, coord: str) -> Any:
    raise NotImplementedError


@Check.register_builtin_check_fn
def has_encoding(data: Any, encoding: dict[str, Any]) -> Any:
    raise NotImplementedError


@Check.register_builtin_check_fn
def cf_standard_name(data: Any, expected_name: str) -> Any:
    raise NotImplementedError


@Check.register_builtin_check_fn
def cf_units(data: Any, expected_units: str) -> Any:
    raise NotImplementedError


@Check.register_builtin_check_fn
def cf_has_standard_names(data: Any, names: tuple[str, ...]) -> Any:
    raise NotImplementedError


@Check.register_builtin_check_fn
def cf_has_cell_methods(data: Any, expected: str) -> Any:
    raise NotImplementedError
