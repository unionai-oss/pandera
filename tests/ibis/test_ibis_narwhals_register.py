"""Tests for register_ibis_backends() narwhals activation.

Requires both ibis and narwhals. Skipped in native-ibis-only sessions;
runs in the tests_narwhals_backend ibis session.
"""

import pytest

pytest.importorskip("narwhals")
import ibis  # noqa: E402


def test_ibis_narwhals_activated_when_opted_in(monkeypatch, request):
    """register_ibis_backends() registers narwhals backends when opt-in is enabled."""
    from pandera.api.ibis.container import (
        DataFrameSchema as IbisDataFrameSchema,
    )
    from pandera.backends.ibis.register import register_ibis_backends
    from pandera.backends.narwhals.container import (
        DataFrameSchemaBackend as NarwhalsDataFrameSchemaBackend,
    )
    from pandera.config import CONFIG

    monkeypatch.setattr(CONFIG, "use_narwhals_backend", True)
    request.addfinalizer(register_ibis_backends.cache_clear)
    register_ibis_backends.cache_clear()
    register_ibis_backends()
    t = ibis.memtable({"a": [1, 2, 3]})
    backend = IbisDataFrameSchema.get_backend(t)
    assert isinstance(backend, NarwhalsDataFrameSchemaBackend)


def test_ibis_backend_is_narwhals():
    """After register_ibis_backends(), ibis.Table uses narwhals DataFrameSchemaBackend."""
    from pandera.api.ibis.container import (
        DataFrameSchema as IbisDataFrameSchema,
    )
    from pandera.backends.ibis.register import register_ibis_backends
    from pandera.backends.narwhals.container import DataFrameSchemaBackend

    register_ibis_backends()
    t = ibis.memtable({"a": [1, 2, 3]})
    backend = IbisDataFrameSchema.get_backend(t)
    assert isinstance(backend, DataFrameSchemaBackend)
