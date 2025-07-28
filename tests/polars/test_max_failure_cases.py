"""Tests for max_failure_cases error message formatting in Polars."""

import polars as pl
import pytest

import pandera.polars as pa
from pandera.config import config_context


def test_max_failure_cases_polars():
    """Test that max_failure_cases limits error message length for polars."""
    
    # Create a DataFrame with many failing values
    df = pl.DataFrame({
        "col1": range(100),  # All values will fail the check
    })
    
    schema = pa.DataFrameSchema({
        "col1": pa.Column(int, pa.Check.greater_than(100))
    })
    
    # Test without limit (default behavior)
    with pytest.raises(pa.errors.SchemaErrors) as exc_info:
        schema.validate(df, lazy=True)
    
    error_message = str(exc_info.value)
    # Polars shows failure cases differently (as a list of dicts)
    assert "failure case examples:" in error_message
    
    # Test with max_failure_cases = 5
    with config_context(max_failure_cases=5):
        with pytest.raises(pa.errors.SchemaErrors) as exc_info:
            schema.validate(df, lazy=True)
        
        error_message = str(exc_info.value)
        # Should show summary of omitted cases
        assert "95 more failure cases (100 total)" in error_message
    
    # Test with max_failure_cases = 1
    with config_context(max_failure_cases=1):
        with pytest.raises(pa.errors.SchemaErrors) as exc_info:
            schema.validate(df, lazy=True)
        
        error_message = str(exc_info.value)
        # Should show summary of omitted cases
        assert "99 more failure cases (100 total)" in error_message


def test_max_failure_cases_polars_edge_cases():
    """Test edge cases for max_failure_cases in polars."""
    
    df = pl.DataFrame({
        "col1": [1, 2, 3],
    })
    
    schema = pa.DataFrameSchema({
        "col1": pa.Column(int, pa.Check.greater_than(10))
    })
    
    # Test with max_failure_cases greater than actual failures
    with config_context(max_failure_cases=10):
        with pytest.raises(pa.errors.SchemaErrors) as exc_info:
            schema.validate(df, lazy=True)
        
        error_message = str(exc_info.value)
        # Should NOT show summary since all cases are shown
        assert "more failure cases" not in error_message
    
    # Test with max_failure_cases = -1 (default, no limit)
    with config_context(max_failure_cases=-1):
        with pytest.raises(pa.errors.SchemaErrors) as exc_info:
            schema.validate(df, lazy=True)
        
        error_message = str(exc_info.value)
        # Should NOT show summary
        assert "more failure cases" not in error_message
