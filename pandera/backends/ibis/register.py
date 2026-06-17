"""Register Ibis backends."""

from functools import lru_cache

import ibis


@lru_cache
def register_ibis_backends(
    check_cls_fqn: str | None = None,
    use_narwhals_backend: bool = False,
):
    """Register backends for Ibis Table types.

    Uses the Narwhals backends when ``use_narwhals_backend`` is ``True`` (from
    ``PANDERA_USE_NARWHALS_BACKEND`` or ``pandera.config.CONFIG``); otherwise
    registers the native Ibis backends.

    Decorated with @lru_cache to prevent duplicate registrations across repeated
    validate() calls. The backend choice is part of the cache key; programmatic
    changes to ``CONFIG.use_narwhals_backend`` after registration trigger
    automatic re-registration via ``pandera.config.set_config``.

    This function is called at schema initialization in the _register_*_backends
    method.
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
        from pandera.api.ibis.components import Column
        from pandera.api.ibis.container import DataFrameSchema
        from pandera.backends.narwhals.checks import NarwhalsCheckBackend
        from pandera.backends.narwhals.components import ColumnBackend
        from pandera.backends.narwhals.container import DataFrameSchemaBackend

        DataFrameSchema.register_backend(
            ibis.Table, DataFrameSchemaBackend, force=True
        )
        Column.register_backend(ibis.Table, ColumnBackend, force=True)
        Check.register_backend(ibis.Table, NarwhalsCheckBackend, force=True)
        Check.register_backend(ibis.Column, NarwhalsCheckBackend, force=True)
        Check.register_backend(nw.LazyFrame, NarwhalsCheckBackend, force=True)
    else:
        import pandera.backends.ibis.builtin_checks  # noqa: F401, I001
        from pandera.api.checks import Check
        from pandera.api.ibis.components import Column
        from pandera.api.ibis.container import DataFrameSchema
        from pandera.backends.ibis.checks import IbisCheckBackend
        from pandera.backends.ibis.components import ColumnBackend  # type: ignore[assignment]
        from pandera.backends.ibis.container import DataFrameSchemaBackend  # type: ignore[assignment]

        DataFrameSchema.register_backend(
            ibis.Table, DataFrameSchemaBackend, force=True
        )
        Column.register_backend(ibis.Table, ColumnBackend, force=True)
        Check.register_backend(ibis.Table, IbisCheckBackend, force=True)
        Check.register_backend(ibis.Column, IbisCheckBackend, force=True)
