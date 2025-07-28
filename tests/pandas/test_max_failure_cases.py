"""Tests for max_failure_cases error message formatting."""

import pandas as pd
import pytest

import pandera.pandas as pa
from pandera import Check, Column, DataFrameSchema
from pandera.config import config_context


def test_default_max_failure_cases():
    """Test that default max_failure_cases is 100."""
    
    # Create a DataFrame with 150 failing values
    df = pd.DataFrame({
        "col1": range(150),  # All values will fail the check
    })
    
    schema = DataFrameSchema({
        "col1": Column(int, Check.greater_than(200))
    })
    
    # Test default behavior (should limit to 100)
    with pytest.raises(pa.errors.SchemaErrors) as exc_info:
        schema.validate(df, lazy=True)
    
    error_message = str(exc_info.value)
    # Should show first 100 values and a summary
    assert "0, 1, 2, 3, 4" in error_message
    assert "99" in error_message  # Should show up to 99 (100th value)
    assert "50 more failure cases (150 total)" in error_message
    assert "149" not in error_message  # Should NOT show the last value


def test_max_failure_cases_pandas():
    """Test that max_failure_cases limits error message length for pandas."""
    
    # Create a DataFrame with many failing values
    df = pd.DataFrame({
        "col1": range(100),  # All values will fail the check
    })
    
    schema = DataFrameSchema({
        "col1": Column(int, Check.greater_than(100))
    })
    
    # Test without limit (default behavior)
    with pytest.raises(pa.errors.SchemaErrors) as exc_info:
        schema.validate(df, lazy=True)
    
    error_message = str(exc_info.value)
    # Should contain all 100 failure cases in the error message
    assert "0, 1, 2, 3, 4, 5, 6, 7, 8, 9" in error_message
    assert "99" in error_message
    
    # Test with max_failure_cases = 5
    with config_context(max_failure_cases=5):
        with pytest.raises(pa.errors.SchemaErrors) as exc_info:
            schema.validate(df, lazy=True)
        
        error_message = str(exc_info.value)
        # Should only show first 5 failure cases
        assert "0, 1, 2, 3, 4" in error_message
        # Should show summary of omitted cases
        assert "95 more failure cases (100 total)" in error_message
        # Should NOT contain later values
        assert "99" not in error_message
    
    # Test with max_failure_cases = 1
    with config_context(max_failure_cases=1):
        with pytest.raises(pa.errors.SchemaErrors) as exc_info:
            schema.validate(df, lazy=True)
        
        error_message = str(exc_info.value)
        # Should only show first failure case
        assert "failure cases: 0" in error_message
        # Should show summary of omitted cases
        assert "99 more failure cases (100 total)" in error_message


def test_max_failure_cases_multiple_checks():
    """Test max_failure_cases with multiple failing checks."""
    
    df = pd.DataFrame({
        "col1": range(50),
        "col2": range(50, 100),
    })
    
    schema = DataFrameSchema({
        "col1": Column(int, [
            Check.greater_than(100),  # All 50 values fail
            Check.less_than(-10),     # All 50 values fail
        ]),
        "col2": Column(int, Check.greater_than(200))  # All 50 values fail
    })
    
    with config_context(max_failure_cases=3):
        with pytest.raises(pa.errors.SchemaErrors) as exc_info:
            schema.validate(df, lazy=True)
        
        error_message = str(exc_info.value)
        
        # Each check should show only 3 failure cases
        assert "0, 1, 2 ... and 47 more failure cases (50 total)" in error_message
        assert "50, 51, 52 ... and 47 more failure cases (50 total)" in error_message


def test_max_failure_cases_edge_cases():
    """Test edge cases for max_failure_cases."""
    
    df = pd.DataFrame({
        "col1": [1, 2, 3],
    })
    
    schema = DataFrameSchema({
        "col1": Column(int, Check.greater_than(10))
    })
    
    # Test with max_failure_cases = 0 (should show no failure cases)
    with config_context(max_failure_cases=0):
        with pytest.raises(pa.errors.SchemaErrors) as exc_info:
            schema.validate(df, lazy=True)
        
        error_message = str(exc_info.value)
        # Should show summary only
        assert "... 3 failure cases" in error_message
    
    # Test with max_failure_cases greater than actual failures
    with config_context(max_failure_cases=10):
        with pytest.raises(pa.errors.SchemaErrors) as exc_info:
            schema.validate(df, lazy=True)
        
        error_message = str(exc_info.value)
        # Should show all 3 failure cases
        assert "1, 2, 3" in error_message
        # Should NOT show summary since all cases are shown
        assert "more failure cases" not in error_message
    
    # Test with max_failure_cases = -1 (default, no limit)
    with config_context(max_failure_cases=-1):
        with pytest.raises(pa.errors.SchemaErrors) as exc_info:
            schema.validate(df, lazy=True)
        
        error_message = str(exc_info.value)
        # Should show all failure cases
        assert "1, 2, 3" in error_message
        assert "more failure cases" not in error_message


def test_max_failure_cases_env_var(monkeypatch):
    """Test that max_failure_cases can be set via environment variable."""
    
    df = pd.DataFrame({
        "col1": range(20),
    })
    
    schema = DataFrameSchema({
        "col1": Column(int, Check.greater_than(50))
    })
    
    # Set environment variable
    monkeypatch.setenv("PANDERA_MAX_FAILURE_CASES", "7")
    
    # Need to reload config to pick up env var
    from pandera import config
    config.CONFIG = config._config_from_env_vars()
    config._CONTEXT_CONFIG = config.copy(config.CONFIG)
    
    with pytest.raises(pa.errors.SchemaErrors) as exc_info:
        schema.validate(df, lazy=True)
    
    error_message = str(exc_info.value)
    # Should show first 7 failure cases
    assert "0, 1, 2, 3, 4, 5, 6" in error_message
    # Should show summary of omitted cases
    assert "13 more failure cases (20 total)" in error_message
    
    # Reset config
    monkeypatch.delenv("PANDERA_MAX_FAILURE_CASES", raising=False)
    config.CONFIG = config._config_from_env_vars()
    config._CONTEXT_CONFIG = config.copy(config.CONFIG)
