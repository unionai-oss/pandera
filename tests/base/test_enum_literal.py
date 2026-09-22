"""Tests for enum and literal introspection helpers."""

import enum
import typing

import pytest

from pandera._enum_literal import (
    enum_categories,
    enum_ordered,
    is_enum_type,
    is_literal_type,
    literal_categories,
    member_descriptions,
    normalize_enum_value,
    value_descriptions,
)


class Department(enum.StrEnum):
    """A documented string enum."""

    billing = "billing"
    """Payment, invoices or subscription issues."""
    technical = "technical"
    """Bugs, outages or integration problems."""
    sales = "sales"
    # deliberately undocumented


class Frustration(enum.IntEnum):
    """A documented int enum."""

    calm = 0
    """Calm, simply stating facts."""
    annoyed = 1
    """Frustrated but civil."""
    angry = 2
    """Very angry, strong language."""


class Plain(enum.Enum):
    """A plain enum whose values differ from its names."""

    first = "a"
    second = "b"


class Multiline(enum.Enum):
    """An enum with a multi-line member docstring."""

    value = "v"
    """First line.

    Second line.
    """


def test_is_enum_type():
    assert is_enum_type(Department)
    assert is_enum_type(Frustration)
    assert not is_enum_type("category")
    assert not is_enum_type(Department.billing)
    assert not is_enum_type(typing.Literal["a"])


def test_is_literal_type():
    assert is_literal_type(typing.Literal["a", "b"])
    assert not is_literal_type(Department)
    assert not is_literal_type(str)


def test_enum_categories_uses_values_not_members():
    # The distinction that matters: a plain Enum's values are not its members.
    assert enum_categories(Plain) == ("a", "b")
    assert enum_categories(Department) == ("billing", "technical", "sales")
    assert enum_categories(Frustration) == (0, 1, 2)


def test_enum_ordered():
    assert enum_ordered(Frustration)
    assert not enum_ordered(Department)
    assert not enum_ordered(Plain)


@pytest.mark.parametrize(
    "literal,expected",
    [
        (typing.Literal["a", "b"], ("a", "b")),
        (typing.Literal[1, 2, 3], (1, 2, 3)),
        # duplicates collapse
        (typing.Literal["a", "b", "a"], ("a", "b")),
    ],
)
def test_literal_categories(literal, expected):
    assert literal_categories(literal) == expected


def test_literal_categories_rejects_heterogeneous():
    with pytest.raises(TypeError, match="mixes value types"):
        literal_categories(typing.Literal["a", 1])


def test_member_descriptions_reads_docstrings():
    assert member_descriptions(Department) == {
        "billing": "Payment, invoices or subscription issues.",
        "technical": "Bugs, outages or integration problems.",
        "sales": None,
    }
    assert member_descriptions(Frustration) == {
        "calm": "Calm, simply stating facts.",
        "annoyed": "Frustrated but civil.",
        "angry": "Very angry, strong language.",
    }


def test_member_descriptions_cleans_multiline_docstrings():
    assert member_descriptions(Multiline) == {
        "value": "First line.\n\nSecond line."
    }


def test_member_descriptions_undocumented_enum():
    assert member_descriptions(Plain) == {"first": None, "second": None}


def test_member_descriptions_without_source_falls_back():
    # A functionally created enum has no retrievable source; the helper must
    # degrade to all-None rather than raise.
    dynamic = enum.Enum("Dynamic", {"a": "a", "b": "b"})
    assert member_descriptions(dynamic) == {"a": None, "b": None}


@pytest.mark.parametrize(
    "value,expected",
    [
        (Plain.first, "a"),  # member
        ("a", "a"),  # value
        ("first", "a"),  # name
        ("unrelated", "unrelated"),  # passthrough
    ],
)
def test_normalize_enum_value(value, expected):
    assert normalize_enum_value(Plain, value) == expected


def test_normalize_enum_value_prefers_value_over_name():
    # ``Ambiguous.a``'s *name* collides with ``Ambiguous.b``'s *value*. A
    # lookup must resolve as a value first so data round-trips unchanged.
    class Ambiguous(enum.Enum):
        a = "z"
        b = "a"

    assert normalize_enum_value(Ambiguous, "a") == "a"


def test_value_descriptions():
    assert value_descriptions(Department) == {
        "billing": "Payment, invoices or subscription issues.",
        "technical": "Bugs, outages or integration problems.",
        "sales": None,
    }
    # Literals have nowhere to carry descriptions.
    assert value_descriptions(
        typing.Literal["a", "b"], categories=("a", "b")
    ) == {"a": None, "b": None}
