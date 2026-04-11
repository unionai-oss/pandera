"""Ibis unit test-specific configuration."""

import sys  # noqa: F401 — available for TEST-01 guard introspection if needed

import pytest

from pandera.config import CONFIG, ValidationDepth, reset_config_context


# TEST-01 guard: tests in tests/ibis/ must not depend on pandera.backends.narwhals.
# The session-scoped fixture below re-registers ibis to win the backend race.


@pytest.fixture(scope="function", autouse=True)
def validation_depth_schema_and_data():
    """Set validation depth to SCHEMA_AND_DATA for ibis unit tests."""
    _validation_depth = CONFIG.validation_depth
    CONFIG.validation_depth = ValidationDepth.SCHEMA_AND_DATA
    try:
        yield
    finally:
        CONFIG.validation_depth = _validation_depth
        reset_config_context()


@pytest.fixture(scope="session", autouse=True)
def _ensure_ibis_backend_registered():
    """TEST-01: ensure the native pandera.backends.ibis backend is active.

    See tests/polars/conftest.py for rationale.
    """
    from pandera.backends.ibis.register import register_ibis_backends
    if hasattr(register_ibis_backends, "cache_clear"):
        register_ibis_backends.cache_clear()
    register_ibis_backends()
    yield
