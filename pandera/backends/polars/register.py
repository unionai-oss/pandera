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

    When the Narwhals backend is opted in, this function additionally
    triggers the cross-backend wiring owned by other native frame
    backends — :func:`pandera.backends.pandas.register.register_pandas_via_narwhals`
    for ``pd.DataFrame`` — so cross-backend validation works regardless of
    whether the user imported ``pandera.pandas`` first. Pyarrow has no
    dedicated pandera register and is set up directly here.

    Decorated with @lru_cache to prevent duplicate registrations across repeated
    validate() calls. The backend choice is fixed at first call — programmatic
    changes to ``CONFIG.use_narwhals_backend`` after registration require
    ``register_polars_backends.cache_clear()`` to take effect.
    """
    from pandera.api.checks import Check
    from pandera.api.polars.components import Column
    from pandera.api.polars.container import DataFrameSchema
    from pandera.config import CONFIG

    if CONFIG.use_narwhals_backend:
        try:
            import narwhals.stable.v1 as nw
        except ImportError as exc:
            raise ImportError(
                "The Narwhals backend is enabled but the 'narwhals' "
                "package is not installed. Install it with: "
                "pip install 'pandera[narwhals]'"
            ) from exc

        import pandera.backends.narwhals.builtin_checks  # noqa: F401
        from pandera.backends.narwhals.checks import NarwhalsCheckBackend
        from pandera.backends.narwhals.components import ColumnBackend
        from pandera.backends.narwhals.container import DataFrameSchemaBackend

        DataFrameSchema.register_backend(pl.LazyFrame, DataFrameSchemaBackend)
        DataFrameSchema.register_backend(pl.DataFrame, DataFrameSchemaBackend)
        Column.register_backend(pl.LazyFrame, ColumnBackend)
        Check.register_backend(pl.LazyFrame, NarwhalsCheckBackend)
        Check.register_backend(nw.LazyFrame, NarwhalsCheckBackend)
        Check.register_backend(nw.DataFrame, NarwhalsCheckBackend)

        # Cross-backend support: pyarrow has no pandera register of its own,
        # so we wire ``pa.Table`` here. Other native frame types
        # (``pd.DataFrame``, ...) are owned by their respective registers
        # and pulled in below so cross-backend validation works regardless
        # of import order.
        try:
            import pyarrow as pa
        except ImportError:
            pass
        else:
            DataFrameSchema.register_backend(pa.Table, DataFrameSchemaBackend)
            Column.register_backend(pa.Table, ColumnBackend)

        from pandera.backends.pandas.register import (
            register_pandas_via_narwhals,
        )

        register_pandas_via_narwhals()
    else:
        import pandera.backends.polars.builtin_checks  # noqa: F401, I001
        from pandera.backends.polars.checks import PolarsCheckBackend
        from pandera.backends.polars.components import ColumnBackend  # type: ignore[assignment]
        from pandera.backends.polars.container import DataFrameSchemaBackend  # type: ignore[assignment]

        DataFrameSchema.register_backend(pl.LazyFrame, DataFrameSchemaBackend)
        DataFrameSchema.register_backend(pl.DataFrame, DataFrameSchemaBackend)
        Column.register_backend(pl.LazyFrame, ColumnBackend)
        Check.register_backend(pl.LazyFrame, PolarsCheckBackend)
