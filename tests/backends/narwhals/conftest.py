"""Shared fixtures for narwhals backend tests."""
import warnings

import pytest
import polars as pl
import narwhals.stable.v1 as nw


@pytest.fixture(autouse=True, scope="module")
def _suppress_narwhals_warning():
    """Initialise narwhals backends and suppress the auto-activation UserWarning.

    Calls register_polars_backends() once per module so that:
    - builtin_checks side-effect runs (populates Dispatcher._function_registry)
    - NarwhalsCheckBackend, ColumnBackend, DataFrameSchemaBackend are registered
    - Tests that call NarwhalsCheckBackend directly do not need to trigger
      schema.validate() first.

    UserWarning is suppressed to keep test output clean.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        from pandera.backends.polars.register import register_polars_backends
        from pandera.backends.ibis.register import register_ibis_backends
        register_polars_backends.cache_clear()
        register_ibis_backends.cache_clear()
        register_polars_backends()
        register_ibis_backends()
        yield


@pytest.fixture(
    params=["polars", "ibis"],
    ids=["polars", "ibis"],
)
def make_narwhals_frame(request):
    """Return a callable that creates an nw.LazyFrame for the given backend."""
    backend = request.param

    def _make(data: dict):
        if backend == "polars":
            return nw.from_native(
                pl.LazyFrame(data), eager_or_interchange_only=False
            )
        elif backend == "ibis":
            import pandas as pd
            import ibis
            return nw.from_native(
                ibis.memtable(pd.DataFrame(data)),
                eager_or_interchange_only=False,
            )

    return _make
