"""Register PyArrow backends."""

from __future__ import annotations

from functools import lru_cache

import pyarrow as pa


@lru_cache
def register_pyarrow_backends(check_cls_fqn: str | None = None):
    """Register backends for ``pyarrow.Table``.

    Unlike polars and ibis — which have hand-written native backends and only
    use narwhals when ``PANDERA_USE_NARWHALS_BACKEND=True`` — pyarrow is served
    exclusively by the narwhals backends, so narwhals is a hard requirement of
    the pyarrow API.

    Decorated with ``@lru_cache`` to prevent duplicate registrations across
    repeated ``validate()`` calls.
    """
    try:
        import narwhals.stable.v1 as nw
    except ImportError as exc:  # pragma: no cover — narwhals is a dependency
        raise ImportError(
            "The pyarrow schema API requires the 'narwhals' package. "
            "Install it with: pip install 'pandera[pyarrow]'"
        ) from exc

    import pandera.backends.narwhals.builtin_checks  # noqa: F401
    from pandera.api.checks import Check
    from pandera.api.pyarrow.components import Column
    from pandera.api.pyarrow.container import DataFrameSchema
    from pandera.backends.narwhals.checks import NarwhalsCheckBackend
    from pandera.backends.narwhals.components import ColumnBackend
    from pandera.backends.narwhals.container import DataFrameSchemaBackend

    DataFrameSchema.register_backend(
        pa.Table, DataFrameSchemaBackend, force=True
    )
    Column.register_backend(pa.Table, ColumnBackend, force=True)
    Check.register_backend(pa.Table, NarwhalsCheckBackend, force=True)
    Check.register_backend(nw.LazyFrame, NarwhalsCheckBackend, force=True)
    Check.register_backend(nw.DataFrame, NarwhalsCheckBackend, force=True)
