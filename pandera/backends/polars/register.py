"""Register polars backends."""

from functools import lru_cache

import polars as pl


@lru_cache
def register_polars_backends(
    check_cls_fqn: str | None = None,
    use_narwhals_backend: bool = False,
):
    """Register backends for Polars frame types.

    Uses the Narwhals backends when ``use_narwhals_backend`` is ``True`` (from
    ``PANDERA_USE_NARWHALS_BACKEND`` or ``pandera.config.CONFIG``); otherwise
    registers the native Polars backends.

    Decorated with @lru_cache to prevent duplicate registrations across repeated
    validate() calls. The backend choice is part of the cache key; programmatic
    changes to ``CONFIG.use_narwhals_backend`` after registration trigger
    automatic re-registration via ``pandera.config.set_config``.
    """
    if use_narwhals_backend:
        try:
            import narwhals.stable.v1 as nw
        except ImportError as exc:
            raise ImportError(
                "The Narwhals backend is enabled but the 'narwhals' "
                "package is not installed. Install it with: "
                "pip install 'pandera[narwhals]'"
            ) from exc

        import pandera.backends.narwhals.builtin_checks  # noqa: F401
        from pandera.api.checks import Check
        from pandera.api.polars.components import Column
        from pandera.api.polars.container import DataFrameSchema
        from pandera.backends.narwhals.checks import NarwhalsCheckBackend
        from pandera.backends.narwhals.components import ColumnBackend
        from pandera.backends.narwhals.container import DataFrameSchemaBackend

        DataFrameSchema.register_backend(
            pl.LazyFrame, DataFrameSchemaBackend, force=True
        )
        DataFrameSchema.register_backend(
            pl.DataFrame, DataFrameSchemaBackend, force=True
        )
        Column.register_backend(pl.LazyFrame, ColumnBackend, force=True)
        Check.register_backend(pl.LazyFrame, NarwhalsCheckBackend, force=True)
        Check.register_backend(nw.LazyFrame, NarwhalsCheckBackend, force=True)
        Check.register_backend(nw.DataFrame, NarwhalsCheckBackend, force=True)
    else:
        import pandera.backends.polars.builtin_checks  # noqa: F401, I001
        from pandera.api.checks import Check
        from pandera.api.polars.components import Column
        from pandera.api.polars.container import DataFrameSchema
        from pandera.backends.polars.checks import PolarsCheckBackend
        from pandera.backends.polars.components import ColumnBackend  # type: ignore[assignment]
        from pandera.backends.polars.container import DataFrameSchemaBackend  # type: ignore[assignment]

        DataFrameSchema.register_backend(
            pl.LazyFrame, DataFrameSchemaBackend, force=True
        )
        DataFrameSchema.register_backend(
            pl.DataFrame, DataFrameSchemaBackend, force=True
        )
        Column.register_backend(pl.LazyFrame, ColumnBackend, force=True)
        Check.register_backend(pl.LazyFrame, PolarsCheckBackend, force=True)
