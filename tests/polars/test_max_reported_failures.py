"""Tests for max_reported_failures error message formatting in Polars."""

import polars as pl
import pytest

import pandera.polars as pa
from pandera.config import config_context


def test_polars_default_shows_all_when_under_100():
    """Test that polars shows all failure cases when under default limit."""
    df = pl.DataFrame({"col1": range(50)})
    schema = pa.DataFrameSchema({
        "col1": pa.Column(int, pa.Check.greater_than(100))
    })
    
    with pytest.raises(pa.errors.SchemaErrors) as exc_info:
        schema.validate(df, lazy=True)
    
    error_message = str(exc_info.value)
    assert "failure case examples:" in error_message
    assert "more failure cases" not in error_message


def test_polars_limits_to_5_failures():
    """Test that setting max_reported_failures to 5 limits polars output."""
    df = pl.DataFrame({"col1": range(100)})
    schema = pa.DataFrameSchema({
        "col1": pa.Column(int, pa.Check.greater_than(100))
    })
    
    with config_context(max_reported_failures=5):
        with pytest.raises(pa.errors.SchemaErrors) as exc_info:
            schema.validate(df, lazy=True)
        
        error_message = str(exc_info.value)
        # Check that we have exactly 5 entries in the failure case examples
        assert "{'col1': 0}" in error_message
        assert "{'col1': 4}" in error_message


def test_polars_shows_omission_count():
    """Test that polars shows correct omission count when truncating."""
    df = pl.DataFrame({"col1": range(100)})
    schema = pa.DataFrameSchema({
        "col1": pa.Column(int, pa.Check.greater_than(100))
    })
    
    with config_context(max_reported_failures=5):
        with pytest.raises(pa.errors.SchemaErrors) as exc_info:
            schema.validate(df, lazy=True)
        
        error_message = str(exc_info.value)
        assert "95 more failure cases (100 total)" in error_message


def test_polars_limits_to_1_failure():
    """Test that setting max_reported_failures to 1 shows only one failure."""
    df = pl.DataFrame({"col1": range(100)})
    schema = pa.DataFrameSchema({
        "col1": pa.Column(int, pa.Check.greater_than(100))
    })
    
    with config_context(max_reported_failures=1):
        with pytest.raises(pa.errors.SchemaErrors) as exc_info:
            schema.validate(df, lazy=True)
        
        error_message = str(exc_info.value)
        assert "{'col1': 0}" in error_message
        assert "99 more failure cases (100 total)" in error_message


def test_polars_zero_shows_summary_only():
    """Test that max_reported_failures=0 shows only summary for polars."""
    df = pl.DataFrame({"col1": [1, 2, 3]})
    schema = pa.DataFrameSchema({
        "col1": pa.Column(int, pa.Check.greater_than(10))
    })
    
    with config_context(max_reported_failures=0):
        with pytest.raises(pa.errors.SchemaErrors) as exc_info:
            schema.validate(df, lazy=True)
        
        error_message = str(exc_info.value)
        assert "... 3 failure cases" in error_message
        assert "{'col1': 1}" not in error_message


def test_polars_exceeding_actual_shows_all():
    """Test that requesting more failures than exist shows all for polars."""
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


def test_polars_unlimited_shows_all():
    """Test that max_reported_failures=-1 shows all failures for polars."""
    df = pl.DataFrame({"col1": [1, 2, 3]})
    schema = pa.DataFrameSchema({
        "col1": pa.Column(int, pa.Check.greater_than(10))
    })
    
    with config_context(max_reported_failures=-1):
        with pytest.raises(pa.errors.SchemaErrors) as exc_info:
            schema.validate(df, lazy=True)
        
        error_message = str(exc_info.value)
        assert "{'col1': 1}" in error_message
        assert "{'col1': 2}" in error_message
        assert "{'col1': 3}" in error_message
        assert "more failure cases" not in error_message


def test_polars_multiple_columns_respect_limit():
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