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

    Decorated with @lru_cache to prevent duplicate registrations across repeated
    validate() calls.

    This function is called at schema initialization in the _register_*_backends
    method.
    """
    from pandera.api.checks import Check
    from pandera.api.ibis.components import Column
    from pandera.api.ibis.container import DataFrameSchema
    from pandera.config import CONFIG

    if CONFIG.use_narwhals_backend:
        import narwhals.stable.v1 as nw

        from pandera.backends.narwhals import (
            builtin_checks,  # noqa — triggers Dispatcher registration
        )
        from pandera.backends.narwhals.checks import NarwhalsCheckBackend
        from pandera.backends.narwhals.components import ColumnBackend
        from pandera.backends.narwhals.container import DataFrameSchemaBackend

        DataFrameSchema.register_backend(ibis.Table, DataFrameSchemaBackend)
        Column.register_backend(ibis.Table, ColumnBackend)
        Check.register_backend(ibis.Table, NarwhalsCheckBackend)
        Check.register_backend(ibis.Column, NarwhalsCheckBackend)
        Check.register_backend(nw.LazyFrame, NarwhalsCheckBackend)
    else:
        from pandera.backends.ibis import builtin_checks  # noqa
        from pandera.backends.ibis.checks import IbisCheckBackend
        from pandera.backends.ibis.components import ColumnBackend  # type: ignore[assignment]
        from pandera.backends.ibis.container import DataFrameSchemaBackend  # type: ignore[assignment]

        DataFrameSchema.register_backend(ibis.Table, DataFrameSchemaBackend)
        Column.register_backend(ibis.Table, ColumnBackend)
        Check.register_backend(ibis.Table, IbisCheckBackend)
        Check.register_backend(ibis.Column, IbisCheckBackend)
