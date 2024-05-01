"""Tests for configuration functions."""

import pytest

from pandera.config import (
    ValidationDepth,
    config_context,
    get_config_context,
    get_config_global,
)


@pytest.mark.parametrize(
    "setting, value, in_ctx_value, post_global_value, post_ctx_value",
    [
        ("validation_enabled", False, False, True, True),
        ("validation_enabled", True, True, True, True),
        # setting validation depth to None will default to SCHEMA_AND_DATA
        # validation depth for the context configuration but retain the None
        # value for the global configuration
        (
            "validation_depth",
            None,
            ValidationDepth.SCHEMA_AND_DATA,
            None,
            ValidationDepth.SCHEMA_AND_DATA,
        ),
        (
            "validation_depth",
            ValidationDepth.SCHEMA_AND_DATA,
            ValidationDepth.SCHEMA_AND_DATA,
            None,
            ValidationDepth.SCHEMA_AND_DATA,
        ),
        (
            "validation_depth",
            ValidationDepth.SCHEMA_ONLY,
            ValidationDepth.SCHEMA_ONLY,
            None,
            ValidationDepth.SCHEMA_AND_DATA,
        ),
        (
            "validation_depth",
            ValidationDepth.DATA_ONLY,
            ValidationDepth.DATA_ONLY,
            None,
            ValidationDepth.SCHEMA_AND_DATA,
        ),
        ("cache_dataframe", True, True, False, False),
        ("cache_dataframe", False, False, False, False),
        ("keep_cached_dataframe", True, True, False, False),
        ("keep_cached_dataframe", False, False, False, False),
    ],
)
def test_config_context(
    setting, value, in_ctx_value, post_global_value, post_ctx_value
):
    with config_context(**{setting: value}):
        config_ctx = get_config_context()
        assert getattr(config_ctx, setting) == in_ctx_value

    config_ctx = get_config_context()
    config_gbl = get_config_global()
    assert getattr(config_ctx, setting) == post_ctx_value
    assert getattr(config_gbl, setting) == post_global_value


@pytest.mark.parametrize(
    "setting, outer_value, inner_value",
    [
        ("validation_enabled", True, False),
        ("validation_enabled", False, True),
        (
            "validation_depth",
            ValidationDepth.SCHEMA_AND_DATA,
            ValidationDepth.SCHEMA_ONLY,
        ),
        (
            "validation_depth",
            ValidationDepth.SCHEMA_AND_DATA,
            ValidationDepth.DATA_ONLY,
        ),
        (
            "validation_depth",
            ValidationDepth.SCHEMA_ONLY,
            ValidationDepth.DATA_ONLY,
        ),
        (
            "validation_depth",
            ValidationDepth.SCHEMA_ONLY,
            ValidationDepth.SCHEMA_AND_DATA,
        ),
        (
            "validation_depth",
            ValidationDepth.DATA_ONLY,
            ValidationDepth.SCHEMA_AND_DATA,
        ),
        (
            "validation_depth",
            ValidationDepth.DATA_ONLY,
            ValidationDepth.SCHEMA_ONLY,
        ),
        ("cache_dataframe", False, True),
        ("cache_dataframe", True, False),
        ("keep_cached_dataframe", False, True),
        ("keep_cached_dataframe", True, False),
    ],
)
def test_nested_config_context(setting, outer_value, inner_value):

    with config_context(**{setting: outer_value}):
        outer_config = get_config_context()
        assert getattr(outer_config, setting) == outer_value

        with config_context(**{setting: inner_value}):
            inner_config = get_config_context()
            assert getattr(inner_config, setting) == inner_value

        outer_config = get_config_context()
        assert getattr(outer_config, setting) == outer_value
