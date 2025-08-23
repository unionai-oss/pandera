"""Tests for max_reported_failures error message formatting."""

import pandas as pd
import pytest

import pandera.pandas as pa
from pandera import Check, Column, DataFrameSchema
from pandera.config import config_context


def test_default_max_reported_failures_is_100():
    """Test that default max_reported_failures is 100."""
    df = pd.DataFrame({"col1": range(150)})
    schema = DataFrameSchema({"col1": Column(int, Check.greater_than(200))})
    
    with pytest.raises(pa.errors.SchemaErrors) as exc_info:
        schema.validate(df, lazy=True)
    
    error_message = str(exc_info.value)
    assert "99" in error_message  # Should show up to index 99 (100 values)
    assert "50 more failure cases (150 total)" in error_message


def test_default_does_not_show_values_beyond_100():
    """Test that default configuration does not show values beyond the 100th."""
    df = pd.DataFrame({"col1": range(150)})
    schema = DataFrameSchema({"col1": Column(int, Check.greater_than(200))})
    
    with pytest.raises(pa.errors.SchemaErrors) as exc_info:
        schema.validate(df, lazy=True)
    
    error_message = str(exc_info.value)
    assert "149" not in error_message


def test_shows_all_failures_when_less_than_default():
    """Test that all failures are shown when count is less than default limit."""
    df = pd.DataFrame({"col1": range(50)})
    schema = DataFrameSchema({"col1": Column(int, Check.greater_than(100))})
    
    with pytest.raises(pa.errors.SchemaErrors) as exc_info:
        schema.validate(df, lazy=True)
    
    error_message = str(exc_info.value)
    assert "49" in error_message  # Last value should be shown
    assert "more failure cases" not in error_message  # No truncation message


def test_max_reported_failures_limits_to_5():
    """Test that setting max_reported_failures to 5 shows only 5 failures."""
    df = pd.DataFrame({"col1": range(100)})
    schema = DataFrameSchema({"col1": Column(int, Check.greater_than(100))})
    
    with config_context(max_reported_failures=5):
        with pytest.raises(pa.errors.SchemaErrors) as exc_info:
            schema.validate(df, lazy=True)
        
        error_message = str(exc_info.value)
        assert "0, 1, 2, 3, 4" in error_message
        # Check that it was truncated
        assert "95 more failure cases" in error_message


def test_max_reported_failures_shows_omission_count():
    """Test that omitted failure count is shown when truncating."""
    df = pd.DataFrame({"col1": range(100)})
    schema = DataFrameSchema({"col1": Column(int, Check.greater_than(100))})
    
    with config_context(max_reported_failures=5):
        with pytest.raises(pa.errors.SchemaErrors) as exc_info:
            schema.validate(df, lazy=True)
        
        error_message = str(exc_info.value)
        assert "95 more failure cases (100 total)" in error_message


def test_max_reported_failures_limits_to_1():
    """Test that setting max_reported_failures to 1 shows only first failure."""
    df = pd.DataFrame({"col1": range(100)})
    schema = DataFrameSchema({"col1": Column(int, Check.greater_than(100))})
    
    with config_context(max_reported_failures=1):
        with pytest.raises(pa.errors.SchemaErrors) as exc_info:
            schema.validate(df, lazy=True)
        
        error_message = str(exc_info.value)
        assert "failure cases: 0" in error_message
        assert "99 more failure cases (100 total)" in error_message


def test_max_reported_failures_zero_shows_summary_only():
    """Test that max_reported_failures=0 shows only summary."""
    df = pd.DataFrame({"col1": [1, 2, 3]})
    schema = DataFrameSchema({"col1": Column(int, Check.greater_than(10))})
    
    with config_context(max_reported_failures=0):
        with pytest.raises(pa.errors.SchemaErrors) as exc_info:
            schema.validate(df, lazy=True)
        
        error_message = str(exc_info.value)
        assert "... 3 failure cases" in error_message
        # Check that no actual values are listed (i.e., no "1, 2, 3" pattern)
        assert "1, 2, 3" not in error_message


def test_max_reported_failures_exceeding_actual_shows_all():
    """Test that requesting more failures than exist shows all without truncation message."""
    df = pd.DataFrame({"col1": [1, 2, 3]})
    schema = DataFrameSchema({"col1": Column(int, Check.greater_than(10))})
    
    with config_context(max_reported_failures=10):
        with pytest.raises(pa.errors.SchemaErrors) as exc_info:
            schema.validate(df, lazy=True)
        
        error_message = str(exc_info.value)
        assert "1, 2, 3" in error_message
        assert "more failure cases" not in error_message


def test_max_reported_failures_unlimited_shows_all():
    """Test that max_reported_failures=-1 shows all failures."""
    df = pd.DataFrame({"col1": [1, 2, 3]})
    schema = DataFrameSchema({"col1": Column(int, Check.greater_than(10))})
    
    with config_context(max_reported_failures=-1):
        with pytest.raises(pa.errors.SchemaErrors) as exc_info:
            schema.validate(df, lazy=True)
        
        error_message = str(exc_info.value)
        assert "1, 2, 3" in error_message
        assert "more failure cases" not in error_message


def test_multiple_checks_each_respect_limit():
    """Test that each check respects the max_reported_failures limit independently."""
    df = pd.DataFrame({
        "col1": range(50),
        "col2": range(50, 100),
    })
    
    schema = DataFrameSchema({
        "col1": Column(int, Check.greater_than(100)),
        "col2": Column(int, Check.greater_than(200))
    })
    
    with config_context(max_reported_failures=3):
        with pytest.raises(pa.errors.SchemaErrors) as exc_info:
            schema.validate(df, lazy=True)
        
        error_message = str(exc_info.value)
        # Both checks should show truncation
        assert "0, 1, 2 ... and 47 more failure cases (50 total)" in error_message
        assert "50, 51, 52 ... and 47 more failure cases (50 total)" in error_message


def test_env_var_sets_max_reported_failures(monkeypatch):
    """Test that PANDERA_MAX_REPORTED_FAILURES environment variable is respected."""
    df = pd.DataFrame({"col1": range(20)})
    schema = DataFrameSchema({"col1": Column(int, Check.greater_than(50))})
    
    monkeypatch.setenv("PANDERA_MAX_REPORTED_FAILURES", "7")
    
    from pandera import config
    config.CONFIG = config._config_from_env_vars()
    config._CONTEXT_CONFIG = config.copy(config.CONFIG)
    
    try:
        with pytest.raises(pa.errors.SchemaErrors) as exc_info:
            schema.validate(df, lazy=True)
        
        error_message = str(exc_info.value)
        assert "0, 1, 2, 3, 4, 5, 6" in error_message
        assert "13 more failure cases (20 total)" in error_message
    finally:
        # Reset config
        monkeypatch.delenv("PANDERA_MAX_REPORTED_FAILURES", raising=False)
        config.CONFIG = config._config_from_env_vars()
        config._CONTEXT_CONFIG = config.copy(config.CONFIG)


def test_env_var_shows_correct_omission_count(monkeypatch):
    """Test that environment variable setting shows correct omission count."""
    df = pd.DataFrame({"col1": range(20)})
    schema = DataFrameSchema({"col1": Column(int, Check.greater_than(50))})
    
    monkeypatch.setenv("PANDERA_MAX_REPORTED_FAILURES", "7")
    
    from pandera import config
    config.CONFIG = config._config_from_env_vars()
    config._CONTEXT_CONFIG = config.copy(config.CONFIG)
    
    try:
        with pytest.raises(pa.errors.SchemaErrors) as exc_info:
            schema.validate(df, lazy=True)
        
        error_message = str(exc_info.value)
        assert "13 more failure cases (20 total)" in error_message
    finally:
        # Reset config
        monkeypatch.delenv("PANDERA_MAX_REPORTED_FAILURES", raising=False)
        config.CONFIG = config._config_from_env_vars()
        config._CONTEXT_CONFIG = config.copy(config.CONFIG)