"""Shared fixtures for narwhals backend tests."""
import pytest
import polars as pl
import narwhals.stable.v1 as nw

from pandera.api.checks import Check


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


@pytest.fixture(autouse=True, scope="module")
def _register_narwhals_check_backend():
    """Register NarwhalsCheckBackend for nw.LazyFrame type and trigger
    builtin_checks side-effect registrations."""
    # Guarded with try/except so dtype tests still collect before Plan 02-02
    # creates the checks.py file; once Plan 02-02 lands, the import succeeds
    # and the backend is registered.
    try:
        from pandera.backends.narwhals.checks import NarwhalsCheckBackend
        Check.register_backend(nw.LazyFrame, NarwhalsCheckBackend)
        # Also register for nw.DataFrame — Ibis tables are wrapped as nw.DataFrame
        # (not nw.LazyFrame) by narwhals, so run_checks on Ibis frames requires
        # a NarwhalsCheckBackend dispatch via DataFrame too.
        Check.register_backend(nw.DataFrame, NarwhalsCheckBackend)
    except ImportError:
        pass
    # Import builtin_checks to trigger CHECK_FUNCTION_REGISTRY side-effect
    # registrations. Guarded with try/except so Wave 1 xfail stubs still
    # collect before Plan 02-03 creates the file.
    try:
        from pandera.backends.narwhals import builtin_checks  # noqa: F401
    except ImportError:
        pass
