"""Tests for max_reported_failures error message formatting in Polars."""

import polars as pl
import pytest

import pandera.polars as pa
from pandera.config import config_context


def test_default_behavior():
    """Test default max_reported_failures behavior for polars."""
    # Test with under 100 failures - should show all
    df = pl.DataFrame({"col1": range(50)})
    schema = pa.DataFrameSchema({
        "col1": pa.Column(int, pa.Check.greater_than(100))
    })
    
    with pytest.raises(pa.errors.SchemaErrors) as exc_info:
        schema.validate(df, lazy=True)
    
    error_message = str(exc_info.value)
    assert "failure case examples:" in error_message
    assert "more failure cases" not in error_message


@pytest.mark.parametrize("limit,expected_values,expected_truncation", [
    (1, [0], "99 more failure cases (100 total)"),
    (5, [0, 4], "95 more failure cases (100 total)"),
])
def test_custom_limits(limit, expected_values, expected_truncation):
    """Test various custom max_reported_failures limits for polars."""
    df = pl.DataFrame({"col1": range(100)})
    schema = pa.DataFrameSchema({
        "col1": pa.Column(int, pa.Check.greater_than(100))
    })
    
    with config_context(max_reported_failures=limit):
        with pytest.raises(pa.errors.SchemaErrors) as exc_info:
            schema.validate(df, lazy=True)
        
        error_message = str(exc_info.value)
        for val in expected_values:
            assert f"{{'col1': {val}}}" in error_message
        assert expected_truncation in error_message


@pytest.mark.parametrize("limit,test_case", [
    (0, "summary_only"),
    (-1, "unlimited"),
])
def test_special_limit_values(limit, test_case):
    """Test special max_reported_failures values for polars."""
    df = pl.DataFrame({"col1": [1, 2, 3]})
    schema = pa.DataFrameSchema({
        "col1": pa.Column(int, pa.Check.greater_than(10))
    })
    
    with config_context(max_reported_failures=limit):
        with pytest.raises(pa.errors.SchemaErrors) as exc_info:
            schema.validate(df, lazy=True)
        
        error_message = str(exc_info.value)
        
        if test_case == "summary_only":
            assert "... 3 failure cases" in error_message
            assert "{'col1': 1}" not in error_message
        else:  # unlimited
            assert "{'col1': 1}" in error_message
            assert "{'col1': 2}" in error_message
            assert "{'col1': 3}" in error_message
            assert "more failure cases" not in error_message


def test_no_truncation_when_under_limit():
    """Test that no truncation occurs when failures are under the limit for polars."""
    df = pl.DataFrame({"col1": [1, 2, 3]})
    schema = pa.DataFrameSchema({
        "col1": pa.Column(int, pa.Check.greater_than(10))
    })
    
    with config_context(max_reported_failures=10):
        with pytest.raises(pa.errors.SchemaErrors) as exc_info:
            schema.validate(df, lazy=True)
        
        error_message = str(exc_info.value)
        assert "{'col1': 1}" in error_message
        assert "{'col1': 2}" in error_message
        assert "{'col1': 3}" in error_message
        assert "more failure cases" not in error_message


def test_multiple_columns_respect_limit():
    """Test that multiple columns each respect the limit in polars."""
    df = pl.DataFrame({
        "col1": range(50),
        "col2": range(50, 100),
    })
    
    schema = pa.DataFrameSchema({
        "col1": pa.Column(int, pa.Check.greater_than(100)),
        "col2": pa.Column(int, pa.Check.greater_than(200))
    })
    
    with config_context(max_reported_failures=3):
        with pytest.raises(pa.errors.SchemaErrors) as exc_info:
            schema.validate(df, lazy=True)
        
        error_message = str(exc_info.value)
        # Both columns should show truncation
        assert "47 more failure cases (50 total)" in error_message