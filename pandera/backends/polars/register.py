"""Register polars backends."""

from functools import lru_cache

import polars as pl


@lru_cache
def register_polars_backends(
    check_cls_fqn: str | None = None,
):
    """Register backends for Polars frame types.

    Uses the Narwhals backends when ``PANDERA_USE_NARWHALS_BACKEND=True`` (or
    ``pandera.config.CONFIG.use_narwhals_backend`` is ``True``); otherwise
    registers the native Polars backends.

    Decorated with @lru_cache to prevent duplicate registrations across repeated
    validate() calls.
    """
    from pandera.api.checks import Check
    from pandera.api.polars.components import Column
    from pandera.api.polars.container import DataFrameSchema
    from pandera.config import CONFIG

    if CONFIG.use_narwhals_backend:
        import narwhals.stable.v1 as nw

        from pandera.backends.narwhals import (
            builtin_checks,  # noqa — triggers Dispatcher registration for NarwhalsData
        )
        from pandera.backends.narwhals.checks import NarwhalsCheckBackend
        from pandera.backends.narwhals.components import ColumnBackend
        from pandera.backends.narwhals.container import DataFrameSchemaBackend

        DataFrameSchema.register_backend(pl.LazyFrame, DataFrameSchemaBackend)
        DataFrameSchema.register_backend(pl.DataFrame, DataFrameSchemaBackend)
        Column.register_backend(pl.LazyFrame, ColumnBackend)
        Check.register_backend(pl.LazyFrame, NarwhalsCheckBackend)
        Check.register_backend(nw.LazyFrame, NarwhalsCheckBackend)
        Check.register_backend(nw.DataFrame, NarwhalsCheckBackend)
    else:
        from pandera.backends.polars import builtin_checks  # noqa
        from pandera.backends.polars.checks import PolarsCheckBackend
        from pandera.backends.polars.components import ColumnBackend  # type: ignore[assignment]
        from pandera.backends.polars.container import DataFrameSchemaBackend  # type: ignore[assignment]

        DataFrameSchema.register_backend(pl.LazyFrame, DataFrameSchemaBackend)
        DataFrameSchema.register_backend(pl.DataFrame, DataFrameSchemaBackend)
        Column.register_backend(pl.LazyFrame, ColumnBackend)
        Check.register_backend(pl.LazyFrame, PolarsCheckBackend)
