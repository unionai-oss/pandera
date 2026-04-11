"""Polars unit test-specific configuration."""

import sys  # noqa: F401 — available for TEST-01 guard introspection if needed

import pytest

from pandera.config import CONFIG, ValidationDepth, reset_config_context


# TEST-01 guard: tests in tests/polars/ must not depend on pandera.backends.narwhals.
# If narwhals backend has been imported by another test earlier in the session,
# the session-scoped fixture below re-registers polars to win the backend race.


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


@pytest.fixture(scope="session", autouse=True)
def _ensure_polars_backend_registered():
    """TEST-01: ensure the native pandera.backends.polars backend is active.

    When narwhals is installed alongside polars (e.g. in a dev environment or
    a future combined CI job), `pandera.backends.narwhals.register` may have
    registered narwhals as the handler for pl.DataFrame / pl.LazyFrame, which
    would shadow the library-native polars backend. This fixture re-registers
    the polars backends at session start so tests/polars/ always exercise the
    native polars backend.
    """
    from pandera.backends.polars.register import register_polars_backends
    if hasattr(register_polars_backends, "cache_clear"):
        register_polars_backends.cache_clear()
    register_polars_backends()
    yield
