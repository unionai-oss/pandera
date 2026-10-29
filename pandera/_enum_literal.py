"""Introspection helpers for ``enum.Enum`` and ``typing.Literal`` annotations.

These are shared by the dtype engines so that every backend derives the same
categories, ordering, and per-member descriptions from the same Python types.
They live in a dependency-free module because the engines import them before
any backend library is guaranteed to be importable.
"""

from __future__ import annotations

import ast
import enum
import inspect
import textwrap
import typing
from collections.abc import Sequence
from functools import cache, lru_cache
from typing import Any, get_args, get_origin

__all__ = [
    "enum_categories",
    "enum_ordered",
    "is_enum_type",
    "is_literal_type",
    "literal_categories",
    "member_descriptions",
    "normalize_enum_value",
]


def is_enum_type(data_type: Any) -> bool:
    """Whether ``data_type`` is an :class:`enum.Enum` subclass."""
    return inspect.isclass(data_type) and issubclass(data_type, enum.Enum)


def is_literal_type(data_type: Any) -> bool:
    """Whether ``data_type`` is a :data:`typing.Literal` alias."""
    return get_origin(data_type) is typing.Literal


def enum_categories(enum_type: type[enum.Enum]) -> tuple[Any, ...]:
    """Return an enum's member *values*, in declaration order.

    Values rather than members: a value is what round-trips through storage and
    serialization, and what every dataframe backend can hold. ``IntEnum`` and
    ``StrEnum`` members compare and hash equal to their values, so this is a
    no-op for them; for a plain ``Enum`` it is the difference between a usable
    categorical dtype and one that rejects its own data.
    """
    return tuple(member.value for member in enum_type)


def enum_ordered(enum_type: type[enum.Enum]) -> bool:
    """Whether an enum implies an ordered categorical.

    ``IntEnum`` and ``IntFlag`` members are comparable, so the categories they
    produce carry a meaningful order; a plain ``Enum`` does not.
    """
    return issubclass(enum_type, (enum.IntEnum, enum.IntFlag))


def literal_categories(literal_type: Any) -> tuple[Any, ...]:
    """Return the arguments of a :data:`typing.Literal`, de-duplicated.

    :raises TypeError: if the literal is empty or mixes value types. A
        heterogeneous literal has no single dtype, and silently picking one
        would discard the rest of the option set.
    """
    args = get_args(literal_type)
    if not args:
        raise TypeError(f"{literal_type} has no literal values.")

    deduped: list[Any] = []
    for arg in args:
        if arg not in deduped:
            deduped.append(arg)

    types = {type(arg) for arg in deduped}
    # bool is a subclass of int, but mixing them is still ambiguous.
    if len(types) > 1:
        names = sorted(type_.__name__ for type_ in types)
        raise TypeError(
            f"Literal {literal_type} mixes value types ({', '.join(names)}). "
            "A Literal used as a data type must be homogeneous."
        )
    return tuple(deduped)


def normalize_enum_value(enum_type: type[enum.Enum], value: Any) -> Any:
    """Map a member, member name, or raw value onto the member's value.

    Used when coercing: a user may reasonably write ``Dept.billing``,
    ``"billing"`` (the value), or ``"billing"`` (the name) and expect all three
    to land in an enum-backed categorical column.
    """
    if isinstance(value, enum_type):
        return value.value
    try:
        if value in enum_type._value2member_map_:
            return value
    except TypeError:  # unhashable value
        return value
    if isinstance(value, str) and value in enum_type.__members__:
        return enum_type[value].value
    return value


@cache
def member_descriptions(
    enum_type: type[enum.Enum],
) -> dict[str, str | None]:
    """Return ``{member_name: docstring}`` for an enum, by reading its source.

    Python discards the bare string literal that follows a member assignment,
    so the only way to recover it is to parse the class source::

        class Department(enum.StrEnum):
            billing = "billing"
            \"\"\"Payment, invoices or subscription issues.\"\"\"

        member_descriptions(Department)
        #> {"billing": "Payment, invoices or subscription issues.", ...}

    Members without a docstring map to ``None``. Returns all-``None`` when the
    source is unavailable — a REPL, a frozen application, or a dynamically
    created enum — so callers can fall back rather than fail.
    """
    descriptions: dict[str, str | None] = {
        name: None for name in enum_type.__members__
    }

    try:
        source = textwrap.dedent(inspect.getsource(enum_type))
    except (OSError, TypeError):
        return descriptions

    try:
        module = ast.parse(source)
    except SyntaxError:  # pragma: no cover - defensive
        return descriptions

    class_def = next(
        (
            node
            for node in module.body
            if isinstance(node, ast.ClassDef)
            and node.name == enum_type.__name__
        ),
        None,
    )
    if class_def is None:  # pragma: no cover - defensive
        return descriptions

    # A member's description is a bare string expression on the statement
    # immediately following its assignment, mirroring how Sphinx and
    # dataclasses-style attribute docs are written.
    pending: str | None = None
    for statement in class_def.body:
        if isinstance(statement, ast.Assign):
            targets = [
                target.id
                for target in statement.targets
                if isinstance(target, ast.Name)
            ]
            pending = targets[0] if len(targets) == 1 else None
        elif isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            pending = statement.target.id
        elif (
            pending is not None
            and isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        ):
            if pending in descriptions:
                descriptions[pending] = inspect.cleandoc(statement.value.value)
            pending = None
        else:
            pending = None

    return descriptions


def value_descriptions(
    data_type: Any,
    categories: Sequence[Any] | None = None,
) -> dict[Any, str | None]:
    """Return ``{category_value: description}`` for an enum or literal type.

    Literals have nowhere to carry descriptions, so every value maps to
    ``None``; enums carry them as member docstrings.
    """
    if is_enum_type(data_type):
        by_name = member_descriptions(data_type)
        return {
            member.value: by_name.get(member.name)
            for member in data_type  # type: ignore[union-attr]
        }
    return {value: None for value in (categories or ())}
