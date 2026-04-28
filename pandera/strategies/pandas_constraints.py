"""Constraint adapters for built-in pandas checks.

Each adapter takes the check's ``statistics`` as kwargs and returns a
:class:`~pandera.strategies.constraints.FieldConstraints` describing
the bounds/membership/regex constraints the check encodes. These are
registered against the built-in checks in
``pandera.backends.pandas.builtin_checks`` via the new
``constraint=`` kwarg on
:func:`~pandera.api.extensions.register_builtin_check`. Once
registered, sibling checks on the same column merge into a single
``FieldConstraints`` and compile to one hypothesis strategy via
:func:`~pandera.strategies.pandas_strategies.compile_field_strategy`,
eliminating ``.filter`` chaining for the built-in check stack.

See ``specs/optimized-strategies.md`` §4.2 for the full table of
adapters.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pandera.strategies.constraints import FieldConstraints

# ----- Equality / inequality -----------------------------------------


def eq_constraint(*, value: Any) -> FieldConstraints:
    return FieldConstraints(eq=value)


def ne_constraint(*, value: Any) -> FieldConstraints:
    return FieldConstraints(notin=frozenset([value]))


# ----- Numeric bounds ------------------------------------------------


def gt_constraint(*, min_value: Any) -> FieldConstraints:
    return FieldConstraints(min_value=min_value, exclude_min=True)


def ge_constraint(*, min_value: Any) -> FieldConstraints:
    return FieldConstraints(min_value=min_value, exclude_min=False)


def lt_constraint(*, max_value: Any) -> FieldConstraints:
    return FieldConstraints(max_value=max_value, exclude_max=True)


def le_constraint(*, max_value: Any) -> FieldConstraints:
    return FieldConstraints(max_value=max_value, exclude_max=False)


def in_range_constraint(
    *,
    min_value: Any,
    max_value: Any,
    include_min: bool = True,
    include_max: bool = True,
) -> FieldConstraints:
    return FieldConstraints(
        min_value=min_value,
        max_value=max_value,
        exclude_min=not include_min,
        exclude_max=not include_max,
    )


# ----- Membership ---------------------------------------------------


def isin_constraint(*, allowed_values: Iterable) -> FieldConstraints:
    return FieldConstraints(isin=frozenset(allowed_values))


def notin_constraint(*, forbidden_values: Iterable) -> FieldConstraints:
    return FieldConstraints(notin=frozenset(forbidden_values))


# ----- Strings ------------------------------------------------------


def str_matches_constraint(*, pattern: str) -> FieldConstraints:
    return FieldConstraints(regex_fullmatch=(str(pattern),))


def str_contains_constraint(*, pattern: str) -> FieldConstraints:
    return FieldConstraints(regex_search=(str(pattern),))


def str_startswith_constraint(*, string: str) -> FieldConstraints:
    return FieldConstraints(regex_fullmatch=(rf"\A(?:{string}).*\Z",))


def str_endswith_constraint(*, string: str) -> FieldConstraints:
    return FieldConstraints(regex_fullmatch=(rf"\A.*(?:{string})\Z",))


def str_length_constraint(
    *,
    min_value: int | None = None,
    max_value: int | None = None,
    exact_value: int | None = None,
) -> FieldConstraints:
    return FieldConstraints(
        str_min_len=min_value,
        str_max_len=max_value,
        str_exact_len=exact_value,
    )


__all__ = [
    "eq_constraint",
    "ne_constraint",
    "gt_constraint",
    "ge_constraint",
    "lt_constraint",
    "le_constraint",
    "in_range_constraint",
    "isin_constraint",
    "notin_constraint",
    "str_matches_constraint",
    "str_contains_constraint",
    "str_startswith_constraint",
    "str_endswith_constraint",
    "str_length_constraint",
]
