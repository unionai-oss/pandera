"""Register narwhals backends, overriding existing polars backend entries."""

from functools import lru_cache


@lru_cache
def register_narwhals_backends(
    check_cls_fqn: str | None = None,
) -> None:
    """Register narwhals backends for Polars frame types.

    Overrides the existing polars backend entries by writing directly into
    BACKEND_REGISTRY. This bypasses the register_backend() guard that would
    otherwise prevent overriding already-registered entries.

    Decorated with lru_cache to prevent duplicate registrations across
    repeated validate() calls.

    Per-library try/except guards allow partial registration when only
    some libraries are installed.

    Called automatically by DataFrameSchemaBackend.validate() when
    get_config_context().use_narwhals_backend is True.
    """
    try:
        import polars as pl

        from pandera.api.checks import Check
        from pandera.api.polars.components import Column
        from pandera.api.polars.container import DataFrameSchema
        from pandera.backends.narwhals.checks import NarwhalsCheckBackend
        from pandera.backends.narwhals.components import ColumnBackend
        from pandera.backends.narwhals.container import DataFrameSchemaBackend

        import narwhals.stable.v1 as nw

        # Direct write — bypasses register_backend() guard:
        # if (cls, type_) not in cls.BACKEND_REGISTRY — would no-op since
        # polars backends are already registered for pl.DataFrame/pl.LazyFrame.
        DataFrameSchema.BACKEND_REGISTRY[
            (DataFrameSchema, pl.DataFrame)
        ] = DataFrameSchemaBackend
        DataFrameSchema.BACKEND_REGISTRY[
            (DataFrameSchema, pl.LazyFrame)
        ] = DataFrameSchemaBackend
        Column.BACKEND_REGISTRY[
            (Column, pl.LazyFrame)
        ] = ColumnBackend
        # Direct write for Check backends — polars Check backend for pl.LazyFrame
        # is registered during polars backend init and must be overridden.
        Check.BACKEND_REGISTRY[
            (Check, pl.LazyFrame)
        ] = NarwhalsCheckBackend
        # Register Check for narwhals-wrapped frames (used inside ColumnBackend.validate)
        Check.BACKEND_REGISTRY[
            (Check, nw.LazyFrame)
        ] = NarwhalsCheckBackend
        Check.BACKEND_REGISTRY[
            (Check, nw.DataFrame)
        ] = NarwhalsCheckBackend
    except ImportError:
        pass
