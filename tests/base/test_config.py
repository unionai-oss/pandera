"""Unit tests for pandera/config.py."""

import pytest

from pandera.config import _CONTEXT_CONFIG, CONFIG, config_context, set_config


def test_set_config_updates_attributes():
    """Test that set_config can update all configuration attributes."""
    original_values = {
        "validation_enabled": CONFIG.validation_enabled,
        "validation_depth": CONFIG.validation_depth,
        "cache_dataframe": CONFIG.cache_dataframe,
        "keep_cached_dataframe": CONFIG.keep_cached_dataframe,
        "use_narwhals_backend": CONFIG.use_narwhals_backend,
        "silenced_warnings": CONFIG.silenced_warnings.copy(),
    }

    try:
        set_config(
            validation_enabled=False,
            validation_depth="DATA_ONLY",
            cache_dataframe=True,
            keep_cached_dataframe=True,
            use_narwhals_backend=True,
            silenced_warnings=["SILENCE_WARNING_PYDANTIC_MODEL"],
        )

        assert CONFIG.validation_enabled is False
        assert CONFIG.validation_depth == "DATA_ONLY"
        assert CONFIG.cache_dataframe is True
        assert CONFIG.keep_cached_dataframe is True
        assert CONFIG.use_narwhals_backend is True
        assert "SILENCE_WARNING_PYDANTIC_MODEL" in CONFIG.silenced_warnings
    finally:
        # Restore original values
        set_config(**original_values)


def test_set_config_invalid_key():
    """Test that invalid configuration keys raise TypeError (not AttributeError)."""
    # With explicit parameters, passing an invalid keyword argument raises TypeError
    with pytest.raises(
        TypeError, match="got an unexpected keyword argument 'invalid_option'"
    ):
        set_config(invalid_option=True)


def test_config_context_honors_new_parameters():
    """Test that config_context properly handles use_narwhals_backend and silenced_warnings parameters."""
    original_use_narwhals = CONFIG.use_narwhals_backend
    original_silenced_warnings = CONFIG.silenced_warnings.copy()

    try:
        with config_context(
            use_narwhals_backend=True,
            silenced_warnings=["SILENCE_WARNING_PYDANTIC_MODEL"],
        ):
            assert _CONTEXT_CONFIG.use_narwhals_backend is True
            assert (
                "SILENCE_WARNING_PYDANTIC_MODEL"
                in _CONTEXT_CONFIG.silenced_warnings
            )
    finally:
        # Restore original values
        CONFIG.use_narwhals_backend = original_use_narwhals
        CONFIG.silenced_warnings = original_silenced_warnings


def test_config_context_isolation():
    """Test that configuration changes are properly isolated within context managers."""
    # Store the original global config values
    original_validation_enabled = CONFIG.validation_enabled
    original_cache_dataframe = CONFIG.cache_dataframe
    original_use_narwhals = CONFIG.use_narwhals_backend
    original_silenced_warnings = CONFIG.silenced_warnings.copy()

    try:
        with config_context(
            validation_enabled=False,
            cache_dataframe=True,
            use_narwhals_backend=True,
            silenced_warnings=["test_warning"],
        ):
            # Check that context config is updated correctly
            assert _CONTEXT_CONFIG.validation_enabled is False
            assert _CONTEXT_CONFIG.cache_dataframe is True
            assert _CONTEXT_CONFIG.use_narwhals_backend is True
            assert "test_warning" in _CONTEXT_CONFIG.silenced_warnings

            # Global config should remain unchanged
            assert CONFIG.validation_enabled == original_validation_enabled
            assert CONFIG.cache_dataframe == original_cache_dataframe
            assert CONFIG.use_narwhals_backend == original_use_narwhals
            assert CONFIG.silenced_warnings == original_silenced_warnings
    finally:
        # Restore original values
        CONFIG.validation_enabled = original_validation_enabled
        CONFIG.cache_dataframe = original_cache_dataframe
        CONFIG.use_narwhals_backend = original_use_narwhals
        CONFIG.silenced_warnings = original_silenced_warnings


def test_config_context_restores_global_config():
    """Test that global CONFIG is correctly restored after context manager exits."""
    original_validation_enabled = CONFIG.validation_enabled
    original_use_narwhals = CONFIG.use_narwhals_backend
    original_silenced_warnings = CONFIG.silenced_warnings.copy()

    try:
        with config_context(
            validation_enabled=False,
            use_narwhals_backend=True,
            silenced_warnings=["test_warning"],
        ):
            pass  # Exit context

        # Global config should be unchanged after context exit
        assert CONFIG.validation_enabled == original_validation_enabled
        assert CONFIG.use_narwhals_backend == original_use_narwhals
        assert CONFIG.silenced_warnings == original_silenced_warnings
    finally:
        CONFIG.validation_enabled = original_validation_enabled
        CONFIG.use_narwhals_backend = original_use_narwhals
        CONFIG.silenced_warnings = original_silenced_warnings


def test_config_context_silenced_warnings():
    """Test that silenced_warnings in config_context properly updates and restores."""
    # Store the original list reference and copy
    original_silenced_warnings_ref = CONFIG.silenced_warnings
    original_silenced_warnings_copy = CONFIG.silenced_warnings.copy()

    try:
        with config_context(silenced_warnings=["test_warning"]):
            # Verify context config is updated correctly
            assert _CONTEXT_CONFIG.silenced_warnings == ["test_warning"]

            # Global config should remain unchanged (we're modifying _CONTEXT_CONFIG, not CONFIG)
            assert CONFIG.silenced_warnings == original_silenced_warnings_copy
            assert CONFIG.silenced_warnings is original_silenced_warnings_ref

        # After context exit, global silenced_warnings should be restored to original state
        assert CONFIG.silenced_warnings == original_silenced_warnings_copy
        assert CONFIG.silenced_warnings is original_silenced_warnings_ref
    finally:
        # Restore to ensure test isolation (this shouldn't change anything)
        pass
