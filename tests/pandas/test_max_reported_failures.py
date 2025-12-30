"""Tests for max_reported_failures error message formatting."""

import pandas as pd
import pytest

import pandera.pandas as pa
from pandera import Check, Column, DataFrameSchema
from pandera.config import config_context


def test_default_behavior():
    """Test default max_reported_failures behavior (limit=100)."""
    df = pd.DataFrame({"col1": range(150)})
    schema = DataFrameSchema({"col1": Column(int, Check.greater_than(200))})

    with pytest.raises(pa.errors.SchemaErrors) as exc_info:
        schema.validate(df, lazy=True)

    error_message = str(exc_info.value)
    # Should show up to index 99 (100 values)
    assert "99" in error_message
    # Should not show values beyond 100
    assert "149" not in error_message
    # Should show truncation message
    assert "50 more failure cases (150 total)" in error_message


@pytest.mark.parametrize(
    "limit,expected_in_message,expected_truncation",
    [
        (1, ["0"], "99 more failure cases (100 total)"),
        (5, ["0, 1, 2, 3, 4"], "95 more failure cases (100 total)"),
        (50, ["49"], "50 more failure cases (100 total)"),
    ],
)
def test_custom_limits(limit, expected_in_message, expected_truncation):
    """Test various custom max_reported_failures limits."""
    df = pd.DataFrame({"col1": range(100)})
    schema = DataFrameSchema({"col1": Column(int, Check.greater_than(100))})

    with config_context(max_reported_failures=limit):
        with pytest.raises(pa.errors.SchemaErrors) as exc_info:
            schema.validate(df, lazy=True)

        error_message = str(exc_info.value)
        for expected in expected_in_message:
            assert str(expected) in error_message
        assert expected_truncation in error_message


@pytest.mark.parametrize(
    "limit,test_case",
    [
        (0, "summary_only"),
        (-1, "unlimited"),
    ],
)
def test_special_limit_values(limit, test_case):
    """Test special max_reported_failures values (0 for summary only, -1 for unlimited)."""
    df = pd.DataFrame({"col1": [1, 2, 3]})
    schema = DataFrameSchema({"col1": Column(int, Check.greater_than(10))})

    with config_context(max_reported_failures=limit):
        with pytest.raises(pa.errors.SchemaErrors) as exc_info:
            schema.validate(df, lazy=True)

        error_message = str(exc_info.value)

        if test_case == "summary_only":
            assert "... 3 failure cases" in error_message
            assert "1, 2, 3" not in error_message
        else:  # unlimited
            assert "1, 2, 3" in error_message
            assert "more failure cases" not in error_message


def test_no_truncation_when_under_limit():
    """Test that no truncation occurs when failures are under the limit."""
    df = pd.DataFrame({"col1": [1, 2, 3]})
    schema = DataFrameSchema({"col1": Column(int, Check.greater_than(10))})

    # Test with default limit (100) which is more than 3 failures
    with pytest.raises(pa.errors.SchemaErrors) as exc_info:
        schema.validate(df, lazy=True)

    error_message = str(exc_info.value)
    assert "1, 2, 3" in error_message
    assert "more failure cases" not in error_message

    # Also test with explicit higher limit
    with config_context(max_reported_failures=10):
        with pytest.raises(pa.errors.SchemaErrors) as exc_info:
            schema.validate(df, lazy=True)

        error_message = str(exc_info.value)
        assert "1, 2, 3" in error_message
        assert "more failure cases" not in error_message


def test_multiple_columns_respect_limit():
    """Test that each column independently respects the max_reported_failures limit."""
    df = pd.DataFrame(
        {
            "col1": range(50),
            "col2": range(50, 100),
        }
    )

    schema = DataFrameSchema(
        {
            "col1": Column(int, Check.greater_than(100)),
            "col2": Column(int, Check.greater_than(200)),
        }
    )

    with config_context(max_reported_failures=3):
        with pytest.raises(pa.errors.SchemaErrors) as exc_info:
            schema.validate(df, lazy=True)

        error_message = str(exc_info.value)
        # Both columns should show truncation
        assert (
            "0, 1, 2 ... and 47 more failure cases (50 total)" in error_message
        )
        assert (
            "50, 51, 52 ... and 47 more failure cases (50 total)"
            in error_message
        )
