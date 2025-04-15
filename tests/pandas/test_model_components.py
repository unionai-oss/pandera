"""Tests individual model components."""

from typing import Any

import pytest

import pandera.pandas as pa
from pandera.engines.pandas_engine import Engine


def test_field_to_column() -> None:
    """Test that Field outputs the correct column options."""
    for flag in ["nullable", "unique", "coerce", "regex"]:
        for value in [True, False]:
            col_kwargs = pa.Field(**{flag: value}).column_properties(  # type: ignore[arg-type]
                pa.DateTime, required=value
            )
            col = pa.Column(**col_kwargs)
            assert col.dtype == Engine.dtype(pa.DateTime)
            assert col.properties[flag] == value
            assert col.required == value


def test_field_to_index() -> None:
    """Test that Field outputs the correct index options."""
    for flag in ["nullable", "unique"]:
        for value in [True, False]:
            index_kwargs = pa.Field(**{flag: value}).index_properties(  # type: ignore[arg-type]
                pa.DateTime
            )
            index = pa.Index(**index_kwargs)
            assert index.dtype == Engine.dtype(pa.DateTime)
            assert getattr(index, flag) == value


def test_field_no_checks() -> None:
    """Test Field without checks."""
    assert not pa.Field().column_properties(str)["checks"]


@pytest.mark.parametrize(
    "arg,value,expected",
    [
        ("eq", 9, pa.Check.equal_to(9)),
        ("ne", 9, pa.Check.not_equal_to(9)),
        ("gt", 9, pa.Check.greater_than(9)),
        ("ge", 9, pa.Check.greater_than_or_equal_to(9)),
        ("lt", 9, pa.Check.less_than(9)),
        ("le", 9, pa.Check.less_than_or_equal_to(9)),
        (
            "in_range",
            {"min_value": 1, "max_value": 9},
            pa.Check.in_range(1, 9),
        ),
        ("isin", [9, "a"], pa.Check.isin([9, "a"])),
        ("notin", [9, "a"], pa.Check.notin([9, "a"])),
        ("str_contains", "a", pa.Check.str_contains("a")),
        ("str_endswith", "a", pa.Check.str_endswith("a")),
        ("str_matches", "a", pa.Check.str_matches("a")),
        (
            "str_length",
            {"min_value": 1, "max_value": 9},
            pa.Check.str_length(1, 9),
        ),
        ("str_startswith", "a", pa.Check.str_startswith("a")),
    ],
)
def test_field_checks(arg: str, value: Any, expected: pa.Check) -> None:
    """Test that all built-in checks are available in a Field."""
    checks = pa.Field(**{arg: value}).column_properties(str)["checks"]
    assert len(checks) == 1
    assert checks[0] == expected
