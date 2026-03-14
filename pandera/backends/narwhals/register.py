"""Register narwhals backends."""

from functools import lru_cache


@lru_cache
def register_narwhals_backends():
    """Register narwhals backends for polars frames.

    This function is idempotent — lru_cache ensures subsequent calls are no-ops.
    It should only be called when the narwhals backend opt-in is active.
    """
    import polars as pl

    from pandera.api.checks import Check
    from pandera.api.polars.components import Column
    from pandera.api.polars.container import DataFrameSchema
    from pandera.backends.narwhals.checks import NarwhalsCheckBackend
    from pandera.backends.narwhals.components import ColumnBackend
    from pandera.backends.narwhals.container import DataFrameSchemaBackend

    import narwhals.stable.v1 as nw

    DataFrameSchema.register_backend(pl.DataFrame, DataFrameSchemaBackend)
    DataFrameSchema.register_backend(pl.LazyFrame, DataFrameSchemaBackend)
    Column.register_backend(pl.LazyFrame, ColumnBackend)
    Check.register_backend(pl.LazyFrame, NarwhalsCheckBackend)
    # Register Check for narwhals-wrapped frames (used inside ColumnBackend.validate)
    Check.register_backend(nw.LazyFrame, NarwhalsCheckBackend)
    Check.register_backend(nw.DataFrame, NarwhalsCheckBackend)
