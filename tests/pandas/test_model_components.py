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
        # in_range with tuple (positional args)
        ("in_range", (0, 1), pa.Check.in_range(0, 1)),
        ("isin", [9, "a"], pa.Check.isin([9, "a"])),
        # isin with tuple - unpacked as positional args
        ("isin", (9, "a"), pa.Check.isin(9, "a")),
        ("notin", [9, "a"], pa.Check.notin([9, "a"])),
        ("str_contains", "a", pa.Check.str_contains("a")),
        ("str_endswith", "a", pa.Check.str_endswith("a")),
        ("str_matches", "a", pa.Check.str_matches("a")),
        # str_length with dict (kwargs style)
        (
            "str_length",
            {"min_value": 1, "max_value": 9},
            pa.Check.str_length(1, 9),
        ),
        # str_length with int (exact length)
        ("str_length", 5, pa.Check.str_length(5)),
        # str_length with single-element tuple (exact length)
        ("str_length", (5,), pa.Check.str_length(5)),
        # str_length with tuple (min, max range)
        ("str_length", (1, 9), pa.Check.str_length(1, 9)),
        ("str_startswith", "a", pa.Check.str_startswith("a")),
    ],
)
def test_field_checks(arg: str, value: Any, expected: pa.Check) -> None:
    """Test that all built-in checks are available in a Field."""
    checks = pa.Field(**{arg: value}).column_properties(str)["checks"]
    assert len(checks) == 1
    assert checks[0] == expected


def test_field_isin_list_and_tuple():
    """Test that Field(isin=list) and Field(isin=tuple) both work correctly."""
    import pandas as pd

    # Test with list
    field_list = pa.Field(isin=[1, 2, 3])
    checks_list = field_list.column_properties(int)["checks"]
    assert len(checks_list) == 1

    # Test with tuple
    field_tuple = pa.Field(isin=(1, 2, 3))
    checks_tuple = field_tuple.column_properties(int)["checks"]
    assert len(checks_tuple) == 1

    # Both should have the same underlying allowed values (as frozensets)
    assert checks_list[0].statistics["allowed_values"] is not None
    assert checks_tuple[0].statistics["allowed_values"] is not None

    # Verify they work in validation
    schema_list = pa.DataFrameSchema(
        {"col": pa.Column(int, checks=checks_list)}
    )
    schema_tuple = pa.DataFrameSchema(
        {"col": pa.Column(int, checks=checks_tuple)}
    )

    valid_df = pd.DataFrame({"col": [1, 2, 3]})
    invalid_df = pd.DataFrame({"col": [1, 2, 4]})

    # Both should pass on valid data
    schema_list.validate(valid_df)
    schema_tuple.validate(valid_df)

    # Both should fail on invalid data
    with pytest.raises(pa.errors.SchemaError):
        schema_list.validate(invalid_df)
    with pytest.raises(pa.errors.SchemaError):
        schema_tuple.validate(invalid_df)


def test_field_in_range_tuple():
    """Test that Field(in_range=(min, max)) works correctly."""
    import pandas as pd

    # Test with tuple (positional args style)
    field_tuple = pa.Field(in_range=(0, 1))
    checks_tuple = field_tuple.column_properties(float)["checks"]
    assert len(checks_tuple) == 1

    # Test with dict (kwargs style)
    field_dict = pa.Field(in_range={"min_value": 0, "max_value": 1})
    checks_dict = field_dict.column_properties(float)["checks"]
    assert len(checks_dict) == 1

    # Both should produce equivalent checks
    assert checks_tuple[0] == checks_dict[0]

    # Verify they work in validation
    schema_tuple = pa.DataFrameSchema(
        {"col": pa.Column(float, checks=checks_tuple)}
    )
    schema_dict = pa.DataFrameSchema(
        {"col": pa.Column(float, checks=checks_dict)}
    )

    valid_df = pd.DataFrame({"col": [0.0, 0.5, 1.0]})
    invalid_df = pd.DataFrame({"col": [0.0, 0.5, 2.0]})

    # Both should pass on valid data
    schema_tuple.validate(valid_df)
    schema_dict.validate(valid_df)

    # Both should fail on invalid data
    with pytest.raises(pa.errors.SchemaError):
        schema_tuple.validate(invalid_df)
    with pytest.raises(pa.errors.SchemaError):
        schema_dict.validate(invalid_df)
