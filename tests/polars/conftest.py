"""Polars unit test-specific configuration."""

import pytest

from pandera.config import CONFIG, ValidationDepth, reset_config_context


@pytest.fixture(scope="function", autouse=True)
def validation_depth_schema_and_data():
    """
    These tests ensure that the validation depth is set to SCHEMA_AND_DATA
    for unit tests.
    """
    _validation_depth = CONFIG.validation_depth
    CONFIG.validation_depth = ValidationDepth.SCHEMA_AND_DATA
    try:
        yield
    finally:
        CONFIG.validation_depth = _validation_depth
        reset_config_context()
