"""Register Ibis backends."""

from functools import lru_cache

import ibis


@lru_cache
def register_ibis_backends(
    check_cls_fqn: str | None = None,
):
    """Register backends for Ibis Table types.

    Uses the Narwhals backends when ``PANDERA_USE_NARWHALS_BACKEND=True`` (or
    ``pandera.config.CONFIG.use_narwhals_backend`` is ``True``); otherwise
    registers the native Ibis backends.

    When the Narwhals backend is opted in, this function additionally
    registers polars frames (``pl.DataFrame`` / ``pl.LazyFrame``) and
    ``pyarrow.Table`` against the ibis schema so they can be validated
    cross-backend, and triggers the
    :func:`pandera.backends.pandas.register.register_pandas_via_narwhals`
    helper so ``pd.DataFrame`` cross-backend validation works regardless of
    whether the user imported ``pandera.pandas`` first.

    Decorated with @lru_cache to prevent duplicate registrations across repeated
    validate() calls. The backend choice is fixed at first call — programmatic
    changes to ``CONFIG.use_narwhals_backend`` after registration require
    ``register_ibis_backends.cache_clear()`` to take effect.

    This function is called at schema initialization in the _register_*_backends
    method.
    """
    from pandera.api.checks import Check
    from pandera.api.ibis.components import Column
    from pandera.api.ibis.container import DataFrameSchema
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

        DataFrameSchema.register_backend(ibis.Table, DataFrameSchemaBackend)
        Column.register_backend(ibis.Table, ColumnBackend)
        Check.register_backend(ibis.Table, NarwhalsCheckBackend)
        Check.register_backend(ibis.Column, NarwhalsCheckBackend)
        Check.register_backend(nw.LazyFrame, NarwhalsCheckBackend)

        # Cross-backend support: polars frames and pyarrow tables have no
        # pandera "destination" beyond their native schemas, so wire them
        # against the ibis schema here. ``pd.DataFrame`` is owned by the
        # pandas register and pulled in below.
        try:
            import polars as pl
        except ImportError:
            pass
        else:
            for pl_cls in (pl.DataFrame, pl.LazyFrame):
                DataFrameSchema.register_backend(
                    pl_cls, DataFrameSchemaBackend
                )
                Column.register_backend(pl_cls, ColumnBackend)

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
        import pandera.backends.ibis.builtin_checks  # noqa: F401, I001
        from pandera.backends.ibis.checks import IbisCheckBackend
        from pandera.backends.ibis.components import ColumnBackend  # type: ignore[assignment]
        from pandera.backends.ibis.container import DataFrameSchemaBackend  # type: ignore[assignment]

        DataFrameSchema.register_backend(ibis.Table, DataFrameSchemaBackend)
        Column.register_backend(ibis.Table, ColumnBackend)
        Check.register_backend(ibis.Table, IbisCheckBackend)
        Check.register_backend(ibis.Column, IbisCheckBackend)
