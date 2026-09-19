"""Tests for programmatic narwhals backend activation via set_config."""

import pytest

pl = pytest.importorskip("polars")
pytest.importorskip("narwhals")


def reset_polars_backend_registry() -> None:
    """Unregister the polars backends, leaving the other backends alone.

    ``BACKEND_REGISTRY`` is a single dict defined on ``BaseSchema`` that every
    backend shares, so ``clear()`` here also unregisters pandas -- and
    ``register_pandas_backends`` is ``lru_cache``d, so those entries never come
    back within the session. Popping per key is why
    ``tests/pyspark/test_pyspark_narwhals_register.py`` does it this way.
    """
    from pandera.api.base.checks import BaseCheck
    from pandera.api.base.parsers import BaseParser
    from pandera.api.base.schema import BaseSchema

    for registry in (
        BaseSchema.BACKEND_REGISTRY,
        BaseCheck.BACKEND_REGISTRY,
        BaseParser.BACKEND_REGISTRY,
    ):
        for key in [
            key
            for key in registry
            if key[0].__module__.startswith("pandera.api.polars")
            or key[1].__module__.startswith("polars.")
        ]:
            registry.pop(key, None)


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
        reset_polars_backend_registry()

        backend = DataFrameSchema.get_backend(pl.DataFrame({"a": [1]}))
        assert isinstance(backend, NarwhalsDataFrameSchemaBackend)
    finally:
        register_polars_backends.cache_clear()
        reset_polars_backend_registry()
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
        reset_polars_backend_registry()

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
        reset_polars_backend_registry()
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
        reset_polars_backend_registry()
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
        reset_polars_backend_registry()
        set_config(use_narwhals_backend=original)
        DataFrameSchema.get_backend(pl.DataFrame({"a": [1]}))


def test_this_module_leaves_other_backends_usable():
    """A pandas schema must still validate after the polars resets above.

    ``BACKEND_REGISTRY`` is one dict on ``BaseSchema`` that every backend
    shares, so a wholesale ``clear()`` also unregisters pandas -- and since
    ``register_pandas_backends`` is ``lru_cache``d, the lazy re-registration
    in ``get_backend`` never puts it back.
    """
    pd = pytest.importorskip("pandas")
    pa_pandas = pytest.importorskip("pandera.pandas")

    schema = pa_pandas.DataFrameSchema({"a": pa_pandas.Column(int)})
    assert schema.validate(pd.DataFrame({"a": [1]})).shape == (1, 1)
