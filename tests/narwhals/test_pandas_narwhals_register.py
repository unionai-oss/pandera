"""Tests for register_pandas_backends() narwhals activation.

Requires both pandas and narwhals. Mirrors
tests/pyspark/test_pyspark_narwhals_register.py and
tests/ibis/test_ibis_narwhals_register.py.
"""

import pandas as pd
import pytest

pytest.importorskip("narwhals")

PANDAS_DATAFRAME_FQN = "pandas.core.frame.DataFrame"


@pytest.fixture
def restore_pandas_registry(request):
    """Snapshot and restore the pd.DataFrame registry entries and cache."""
    from pandera._pandas_deprecated import (
        DataFrameSchema as DataFrameSchemaDeprecated,
    )
    from pandera.api.checks import Check
    from pandera.api.pandas.components import Column
    from pandera.api.pandas.container import DataFrameSchema
    from pandera.backends.pandas.register import register_pandas_backends

    registry_keys = [
        (DataFrameSchema, (DataFrameSchema, pd.DataFrame)),
        (
            DataFrameSchemaDeprecated,
            (DataFrameSchemaDeprecated, pd.DataFrame),
        ),
        (Column, (Column, pd.DataFrame)),
        (Check, (Check, pd.DataFrame)),
    ]
    saved = [
        (cls, key, cls.BACKEND_REGISTRY.pop(key, None))
        for cls, key in registry_keys
    ]

    def restore():
        register_pandas_backends.cache_clear()
        for cls, key, backend in saved:
            if backend is None:
                cls.BACKEND_REGISTRY.pop(key, None)
            else:
                cls.BACKEND_REGISTRY[key] = backend

    request.addfinalizer(restore)
    register_pandas_backends.cache_clear()


def test_pandas_narwhals_activated_when_opted_in(restore_pandas_registry):
    """register_pandas_backends() registers narwhals backends when opted in."""
    import narwhals.stable.v1 as nw

    from pandera._pandas_deprecated import (
        DataFrameSchema as DataFrameSchemaDeprecated,
    )
    from pandera.api.checks import Check
    from pandera.api.pandas.components import Column
    from pandera.api.pandas.container import DataFrameSchema
    from pandera.backends.narwhals.checks import NarwhalsCheckBackend
    from pandera.backends.narwhals.components import (
        ColumnBackend as NarwhalsColumnBackend,
    )
    from pandera.backends.narwhals.container import (
        DataFrameSchemaBackend as NarwhalsDataFrameSchemaBackend,
    )
    from pandera.backends.pandas.checks import PandasCheckBackend
    from pandera.backends.pandas.register import register_pandas_backends

    register_pandas_backends(PANDAS_DATAFRAME_FQN, use_narwhals_backend=True)

    backend = DataFrameSchema.get_backend(check_type=pd.DataFrame)
    assert isinstance(backend, NarwhalsDataFrameSchemaBackend)

    deprecated_backend = DataFrameSchemaDeprecated.get_backend(
        check_type=pd.DataFrame
    )
    assert isinstance(deprecated_backend, NarwhalsDataFrameSchemaBackend)

    column_backend = Column.get_backend(check_type=pd.DataFrame)
    assert isinstance(column_backend, NarwhalsColumnBackend)

    # Narwhals-wrapped frames dispatch to the narwhals check backend; the
    # native pd.DataFrame entry stays on the native pandas check backend so
    # direct Check calls on native frames are unchanged.
    assert (
        Check.BACKEND_REGISTRY[(Check, nw.LazyFrame)] is NarwhalsCheckBackend
    )
    assert (
        Check.BACKEND_REGISTRY[(Check, nw.DataFrame)] is NarwhalsCheckBackend
    )
    assert Check.BACKEND_REGISTRY[(Check, pd.DataFrame)] is PandasCheckBackend


def test_pandas_native_unchanged_when_flag_off(
    monkeypatch, restore_pandas_registry
):
    """register_pandas_backends() registers native backends when opted out."""
    from pandera.api.pandas.container import DataFrameSchema
    from pandera.backends.pandas.container import (
        DataFrameSchemaBackend as NativeBackend,
    )
    from pandera.backends.pandas.register import register_pandas_backends
    from pandera.config import CONFIG

    # get_backend() re-invokes registration reading CONFIG, so the flag must
    # be off there as well.
    monkeypatch.setattr(CONFIG, "use_narwhals_backend", False)
    register_pandas_backends(PANDAS_DATAFRAME_FQN, use_narwhals_backend=False)
    backend = DataFrameSchema.get_backend(check_type=pd.DataFrame)
    assert isinstance(backend, NativeBackend)


def test_pandas_series_and_index_backends_stay_native(
    restore_pandas_registry,
):
    """SeriesSchema/Index/MultiIndex stay on the native pandas backends."""
    from pandera.api.pandas.array import SeriesSchema
    from pandera.api.pandas.components import Index, MultiIndex
    from pandera.backends.pandas.array import SeriesSchemaBackend
    from pandera.backends.pandas.components import (
        IndexBackend,
        MultiIndexBackend,
    )
    from pandera.backends.pandas.register import register_pandas_backends

    register_pandas_backends(PANDAS_DATAFRAME_FQN, use_narwhals_backend=True)

    assert isinstance(
        SeriesSchema.get_backend(check_type=pd.Series), SeriesSchemaBackend
    )
    assert isinstance(Index.get_backend(check_type=pd.DataFrame), IndexBackend)
    assert isinstance(
        MultiIndex.get_backend(check_type=pd.DataFrame), MultiIndexBackend
    )


def test_pandas_register_is_idempotent():
    """Calling register_pandas_backends() twice does not raise or corrupt state."""
    from pandera.backends.pandas.register import register_pandas_backends
    from pandera.config import CONFIG

    register_pandas_backends(
        PANDAS_DATAFRAME_FQN,
        use_narwhals_backend=CONFIG.use_narwhals_backend,
    )
    register_pandas_backends(
        PANDAS_DATAFRAME_FQN,
        use_narwhals_backend=CONFIG.use_narwhals_backend,
    )


def test_re_register_pandas_backends_replays_fqns(
    monkeypatch, restore_pandas_registry
):
    """re_register_pandas_backends() replays registrations with the new flag."""
    from pandera.api.pandas.container import DataFrameSchema
    from pandera.backends.narwhals.container import (
        DataFrameSchemaBackend as NarwhalsDataFrameSchemaBackend,
    )
    from pandera.backends.pandas.container import (
        DataFrameSchemaBackend as NativeBackend,
    )
    from pandera.backends.pandas.register import (
        re_register_pandas_backends,
        register_pandas_backends,
    )
    from pandera.config import CONFIG

    # get_backend() re-invokes registration reading CONFIG, so keep CONFIG in
    # sync with each re-registration below.
    monkeypatch.setattr(CONFIG, "use_narwhals_backend", False)
    register_pandas_backends(PANDAS_DATAFRAME_FQN, use_narwhals_backend=False)
    assert isinstance(
        DataFrameSchema.get_backend(check_type=pd.DataFrame), NativeBackend
    )

    monkeypatch.setattr(CONFIG, "use_narwhals_backend", True)
    re_register_pandas_backends(use_narwhals_backend=True)
    assert isinstance(
        DataFrameSchema.get_backend(check_type=pd.DataFrame),
        NarwhalsDataFrameSchemaBackend,
    )

    # Toggling back re-registers the native backend. Registry entries from
    # the narwhals registration must be cleared first — mirroring what
    # set_config() does via reregister_narwhals_compatible_backends().
    from pandera.backends.narwhals.register import (
        clear_narwhals_compatible_backend_registry,
    )

    monkeypatch.setattr(CONFIG, "use_narwhals_backend", False)
    clear_narwhals_compatible_backend_registry()
    re_register_pandas_backends(use_narwhals_backend=False)
    assert isinstance(
        DataFrameSchema.get_backend(check_type=pd.DataFrame), NativeBackend
    )


def test_set_config_reregisters_pandas_backends(restore_pandas_registry):
    """set_config(use_narwhals_backend=...) swaps registered pandas backends."""
    import pandera
    from pandera.api.pandas.container import DataFrameSchema
    from pandera.backends.narwhals.container import (
        DataFrameSchemaBackend as NarwhalsDataFrameSchemaBackend,
    )
    from pandera.backends.pandas.container import (
        DataFrameSchemaBackend as NativeBackend,
    )
    from pandera.backends.pandas.register import register_pandas_backends
    from pandera.config import CONFIG

    original_flag = CONFIG.use_narwhals_backend

    try:
        pandera.set_config(use_narwhals_backend=False)
        register_pandas_backends(
            PANDAS_DATAFRAME_FQN, use_narwhals_backend=False
        )
        assert isinstance(
            DataFrameSchema.get_backend(check_type=pd.DataFrame),
            NativeBackend,
        )

        with pytest.warns(UserWarning, match="Re-registered"):
            pandera.set_config(use_narwhals_backend=True)
        assert isinstance(
            DataFrameSchema.get_backend(check_type=pd.DataFrame),
            NarwhalsDataFrameSchemaBackend,
        )

        with pytest.warns(UserWarning, match="Re-registered"):
            pandera.set_config(use_narwhals_backend=False)
        assert isinstance(
            DataFrameSchema.get_backend(check_type=pd.DataFrame),
            NativeBackend,
        )
    finally:
        pandera.set_config(use_narwhals_backend=original_flag)
