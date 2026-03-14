"""Shared fixtures for narwhals backend tests."""
import warnings

import pytest
import polars as pl
import narwhals.stable.v1 as nw


@pytest.fixture(autouse=True, scope="module")
def _suppress_narwhals_warning():
    """Suppress the narwhals auto-activation UserWarning during tests."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
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
