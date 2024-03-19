"""Polars unit test-specific configuration."""

import pytest

from pandera.config import CONFIG, ValidationDepth


@pytest.fixture(scope="function", autouse=True)
def set_validation_depth():
    """
    These tests ensure that the validation depth is set to SCHEMA_AND_DATA
    for unit tests.
    """
    _validation_depth = CONFIG.validation_depth
    CONFIG.validation_depth = ValidationDepth.SCHEMA_AND_DATA
    yield
    CONFIG.validation_depth = _validation_depth
