"""Tests for register_pyspark_backends() narwhals activation.

Requires both pyspark and narwhals. Skipped in native-pyspark-only sessions;
runs in the tests_narwhals_backend pyspark session.
"""

import pytest

pytest.importorskip("narwhals")


def test_pyspark_narwhals_activated_when_opted_in(monkeypatch, request):
    """register_pyspark_backends() registers narwhals backends when opt-in is enabled."""
    import pyspark.sql as pyspark_sql

    from pandera.api.pyspark.container import (
        DataFrameSchema as PySparkDataFrameSchema,
    )
    from pandera.backends.narwhals.container import (
        DataFrameSchemaBackend as NarwhalsDataFrameSchemaBackend,
    )
    from pandera.backends.pyspark.register import register_pyspark_backends
    from pandera.config import CONFIG

    registry_key = (PySparkDataFrameSchema, pyspark_sql.DataFrame)
    saved = PySparkDataFrameSchema.BACKEND_REGISTRY.pop(registry_key, None)
    request.addfinalizer(register_pyspark_backends.cache_clear)
    if saved is not None:
        request.addfinalizer(
            lambda: PySparkDataFrameSchema.BACKEND_REGISTRY.update(
                {registry_key: saved}
            )
        )

    monkeypatch.setattr(CONFIG, "use_narwhals_backend", True)
    register_pyspark_backends.cache_clear()
    register_pyspark_backends()
    backend = PySparkDataFrameSchema.get_backend(check_type=pyspark_sql.DataFrame)
    assert isinstance(backend, NarwhalsDataFrameSchemaBackend)


def test_pyspark_native_unchanged_when_flag_off(monkeypatch, request):
    """register_pyspark_backends() registers native backends when opt-in is disabled."""
    import pyspark.sql as pyspark_sql

    from pandera.api.pyspark.container import (
        DataFrameSchema as PySparkDataFrameSchema,
    )
    from pandera.backends.pyspark.container import (
        DataFrameSchemaBackend as NativeBackend,
    )
    from pandera.backends.pyspark.register import register_pyspark_backends
    from pandera.config import CONFIG

    registry_key = (PySparkDataFrameSchema, pyspark_sql.DataFrame)
    saved = PySparkDataFrameSchema.BACKEND_REGISTRY.pop(registry_key, None)
    request.addfinalizer(register_pyspark_backends.cache_clear)
    if saved is not None:
        request.addfinalizer(
            lambda: PySparkDataFrameSchema.BACKEND_REGISTRY.update(
                {registry_key: saved}
            )
        )

    monkeypatch.setattr(CONFIG, "use_narwhals_backend", False)
    register_pyspark_backends.cache_clear()
    register_pyspark_backends()
    backend = PySparkDataFrameSchema.get_backend(check_type=pyspark_sql.DataFrame)
    assert isinstance(backend, NativeBackend)


def test_pyspark_connect_narwhals_activated_when_opted_in(monkeypatch, request):
    """register_pyspark_backends() registers narwhals backends for pyspark_connect.DataFrame."""
    import pyspark
    from packaging import version

    if version.parse(pyspark.__version__) < version.parse("3.4"):
        pytest.skip("pyspark.sql.connect requires pyspark >= 3.4")

    try:
        from pyspark.sql.connect import dataframe as pyspark_connect
    except Exception:
        pytest.skip(
            "pyspark.sql.connect dependencies (e.g. grpcio-status) not installed"
        )

    from pandera.api.pyspark.container import (
        DataFrameSchema as PySparkDataFrameSchema,
    )
    from pandera.backends.narwhals.container import (
        DataFrameSchemaBackend as NarwhalsDataFrameSchemaBackend,
    )
    from pandera.backends.pyspark.register import register_pyspark_backends
    from pandera.config import CONFIG

    registry_key = (PySparkDataFrameSchema, pyspark_connect.DataFrame)
    saved = PySparkDataFrameSchema.BACKEND_REGISTRY.pop(registry_key, None)
    request.addfinalizer(register_pyspark_backends.cache_clear)
    if saved is not None:
        request.addfinalizer(
            lambda: PySparkDataFrameSchema.BACKEND_REGISTRY.update(
                {registry_key: saved}
            )
        )

    monkeypatch.setattr(CONFIG, "use_narwhals_backend", True)
    register_pyspark_backends.cache_clear()
    register_pyspark_backends()
    backend = PySparkDataFrameSchema.get_backend(
        check_type=pyspark_connect.DataFrame
    )
    assert isinstance(backend, NarwhalsDataFrameSchemaBackend)


def test_pyspark_register_is_idempotent():
    """Calling register_pyspark_backends() twice does not raise or corrupt state."""
    from pandera.backends.pyspark.register import register_pyspark_backends

    register_pyspark_backends()
    register_pyspark_backends()
