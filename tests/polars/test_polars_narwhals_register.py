"""Tests for programmatic narwhals backend activation via set_config."""

import pytest

pl = pytest.importorskip("polars")
pytest.importorskip("narwhals")


def test_import_polars_does_not_register_backends():
    """import pandera.polars must not eagerly register validation backends."""
    import importlib

    from pandera.backends.polars.register import register_polars_backends

    register_polars_backends.cache_clear()
    importlib.reload(importlib.import_module("pandera.polars"))
    assert register_polars_backends.cache_info().currsize == 0


def test_set_config_before_import_uses_narwhals_backend():
    """set_config before first schema use registers narwhals backends."""
    from pandera.api.polars.container import DataFrameSchema
    from pandera.backends.narwhals.container import (
        DataFrameSchemaBackend as NarwhalsDataFrameSchemaBackend,
    )
    from pandera.backends.polars.register import register_polars_backends
    from pandera.config import CONFIG, set_config

    original = CONFIG.use_narwhals_backend
    try:
        set_config(use_narwhals_backend=True)
        register_polars_backends.cache_clear()
        DataFrameSchema.BACKEND_REGISTRY.clear()

        backend = DataFrameSchema.get_backend(pl.DataFrame({"a": [1]}))
        assert isinstance(backend, NarwhalsDataFrameSchemaBackend)
    finally:
        register_polars_backends.cache_clear()
        DataFrameSchema.BACKEND_REGISTRY.clear()
        set_config(use_narwhals_backend=original)
        DataFrameSchema.get_backend(pl.DataFrame({"a": [1]}))


def test_set_config_after_import_switches_to_narwhals_backend():
    """set_config after import re-registers backends (test.py scenario)."""
    import pandera.polars as pa
    from pandera.backends.narwhals.container import (
        DataFrameSchemaBackend as NarwhalsDataFrameSchemaBackend,
    )
    from pandera.backends.polars.container import (
        DataFrameSchemaBackend as NativeDataFrameSchemaBackend,
    )
    from pandera.backends.polars.register import register_polars_backends
    from pandera.config import CONFIG, set_config

    original = CONFIG.use_narwhals_backend
    try:
        set_config(use_narwhals_backend=False)
        register_polars_backends.cache_clear()
        pa.DataFrameSchema.BACKEND_REGISTRY.clear()

        schema = pa.DataFrameSchema({"name": pa.Column(str)})
        native_backend = pa.DataFrameSchema.get_backend(
            pl.DataFrame({"name": ["a"]})
        )
        assert isinstance(native_backend, NativeDataFrameSchemaBackend)

        with pytest.warns(UserWarning, match="Re-registered pandera backends"):
            pa.config.set_config(use_narwhals_backend=True)

        narwhals_backend = pa.DataFrameSchema.get_backend(
            pl.DataFrame({"name": ["a"]})
        )
        assert isinstance(narwhals_backend, NarwhalsDataFrameSchemaBackend)
        assert schema.validate(pl.DataFrame({"name": ["John"]})).equals(
            pl.DataFrame({"name": ["John"]})
        )
    finally:
        register_polars_backends.cache_clear()
        pa.DataFrameSchema.BACKEND_REGISTRY.clear()
        set_config(use_narwhals_backend=original)
        pa.DataFrameSchema.get_backend(pl.DataFrame({"name": ["a"]}))


def test_set_config_toggles_native_and_narwhals():
    """set_config can switch between native and narwhals backends in-process."""
    from pandera.api.polars.container import DataFrameSchema
    from pandera.backends.narwhals.container import (
        DataFrameSchemaBackend as NarwhalsDataFrameSchemaBackend,
    )
    from pandera.backends.polars.container import (
        DataFrameSchemaBackend as NativeDataFrameSchemaBackend,
    )
    from pandera.backends.polars.register import register_polars_backends
    from pandera.config import CONFIG, set_config

    original = CONFIG.use_narwhals_backend
    try:
        set_config(use_narwhals_backend=False)
        register_polars_backends.cache_clear()
        DataFrameSchema.BACKEND_REGISTRY.clear()
        DataFrameSchema.get_backend(pl.DataFrame({"a": [1]}))
        assert isinstance(
            DataFrameSchema.get_backend(pl.DataFrame({"a": [1]})),
            NativeDataFrameSchemaBackend,
        )

        with pytest.warns(UserWarning, match="Re-registered pandera backends"):
            set_config(use_narwhals_backend=True)
        assert isinstance(
            DataFrameSchema.get_backend(pl.DataFrame({"a": [1]})),
            NarwhalsDataFrameSchemaBackend,
        )

        with pytest.warns(UserWarning, match="Re-registered pandera backends"):
            set_config(use_narwhals_backend=False)
        assert isinstance(
            DataFrameSchema.get_backend(pl.DataFrame({"a": [1]})),
            NativeDataFrameSchemaBackend,
        )
    finally:
        register_polars_backends.cache_clear()
        DataFrameSchema.BACKEND_REGISTRY.clear()
        set_config(use_narwhals_backend=original)
        DataFrameSchema.get_backend(pl.DataFrame({"a": [1]}))
