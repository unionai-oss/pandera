"""Register DataFusion backends."""

from __future__ import annotations

from functools import lru_cache

import datafusion


@lru_cache
def register_datafusion_backends(check_cls_fqn: str | None = None):
    """Register backends for ``datafusion.DataFrame``.

    Like pyarrow, DataFusion has no hand-written native backend: validation
    is served exclusively by the narwhals backends. Narwhals reaches a
    ``datafusion.DataFrame`` through the ``narwhals-datafusion`` plugin
    (discovered via the ``narwhals.plugins`` entry point), so both packages
    are hard requirements of the DataFusion API.

    Decorated with ``@lru_cache`` to prevent duplicate registrations across
    repeated ``validate()`` calls.
    """
    try:
        import narwhals.stable.v1 as nw
    except ImportError as exc:  # pragma: no cover — narwhals is a dependency
        raise ImportError(
            "The DataFusion schema API requires the 'narwhals' package. "
            "Install it with: pip install 'pandera[datafusion]'"
        ) from exc

    try:
        import narwhals_datafusion  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The DataFusion schema API requires the 'narwhals-datafusion' "
            "plugin so that narwhals can wrap datafusion DataFrames. "
            "Install it with: pip install 'pandera[datafusion]'"
        ) from exc

    import pandera.backends.narwhals.builtin_checks  # noqa: F401
    from pandera.api.checks import Check
    from pandera.api.datafusion.components import Column
    from pandera.api.datafusion.container import DataFrameSchema
    from pandera.backends.narwhals.checks import NarwhalsCheckBackend
    from pandera.backends.narwhals.components import ColumnBackend
    from pandera.backends.narwhals.container import DataFrameSchemaBackend

    DataFrameSchema.register_backend(
        datafusion.DataFrame, DataFrameSchemaBackend, force=True
    )
    Column.register_backend(datafusion.DataFrame, ColumnBackend, force=True)
    Check.register_backend(
        datafusion.DataFrame, NarwhalsCheckBackend, force=True
    )
    Check.register_backend(nw.LazyFrame, NarwhalsCheckBackend, force=True)
    Check.register_backend(nw.DataFrame, NarwhalsCheckBackend, force=True)
