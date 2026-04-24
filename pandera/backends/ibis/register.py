"""Register Ibis backends."""

from functools import lru_cache

import ibis


def _optional_native_frame_types() -> list[type]:
    """Collect optional narwhals-supported native frame classes.

    Each entry is registered against the ibis schema/backend dispatch so
    users can validate, e.g., a ``pd.DataFrame`` or a ``pyspark.sql.DataFrame``
    with an ibis schema. The narwhals backend wraps the native input
    internally, so the same backend implementations work uniformly across
    native types. Imports are best-effort — missing packages are silently
    skipped.
    """
    classes: list[type] = []
    try:
        import pandas as pd

        classes.append(pd.DataFrame)
    except ImportError:
        pass
    try:
        import polars as pl

        classes.extend([pl.DataFrame, pl.LazyFrame])
    except ImportError:
        pass
    try:
        import pyarrow as pa

        classes.append(pa.Table)
    except ImportError:
        pass
    try:
        import pyspark.sql

        classes.append(pyspark.sql.DataFrame)
    except ImportError:
        pass
    # Spark Connect (pyspark>=3.4) ships its own ``DataFrame`` class which
    # narwhals dispatches as ``Implementation.PYSPARK_CONNECT``. Registering
    # both keeps cross-backend coverage in step with the legacy pyspark
    # register at ``pandera.backends.pyspark.register``.
    try:
        from pyspark.sql.connect import dataframe as pyspark_connect

        classes.append(pyspark_connect.DataFrame)
    except ImportError:
        pass
    return classes


@lru_cache
def register_ibis_backends(
    check_cls_fqn: str | None = None,
):
    """Register backends for Ibis Table types.

    Uses the Narwhals backends when ``PANDERA_USE_NARWHALS_BACKEND=True`` (or
    ``pandera.config.CONFIG.use_narwhals_backend`` is ``True``); otherwise
    registers the native Ibis backends.

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

        # Cross-backend support: enable validating any narwhals-supported
        # native frame (pandas, polars, pyarrow, pyspark.sql.DataFrame, ...)
        # against an ibis schema. The narwhals backend wraps the native input
        # via ``nw.from_native`` and operates on a Narwhals LazyFrame
        # internally, so the same backend implementations work uniformly
        # across native types.
        #
        # IMPORTANT: We deliberately do NOT register the cross-backend native
        # frames against ``Check``. ``Check.register_backend`` is first-write-
        # wins, and the legacy pandas/pyspark backends already claim
        # ``pd.DataFrame`` / ``pyspark.sql.DataFrame`` — letting both compete
        # would silently break each other's tests. Custom dataframe-level
        # checks always receive a Narwhals frame inside the narwhals backend
        # (``check_obj`` is wrapped to ``nw.LazyFrame`` before dispatch), so
        # dispatch by the registered ``nw.LazyFrame`` entry is sufficient.
        for native_cls in _optional_native_frame_types():
            DataFrameSchema.register_backend(
                native_cls, DataFrameSchemaBackend
            )
            Column.register_backend(native_cls, ColumnBackend)
    else:
        import pandera.backends.ibis.builtin_checks  # noqa: F401, I001
        from pandera.backends.ibis.checks import IbisCheckBackend
        from pandera.backends.ibis.components import ColumnBackend  # type: ignore[assignment]
        from pandera.backends.ibis.container import DataFrameSchemaBackend  # type: ignore[assignment]

        DataFrameSchema.register_backend(ibis.Table, DataFrameSchemaBackend)
        Column.register_backend(ibis.Table, ColumnBackend)
        Check.register_backend(ibis.Table, IbisCheckBackend)
        Check.register_backend(ibis.Column, IbisCheckBackend)
