"""Register pyspark backends."""

from functools import lru_cache

import pyspark
import pyspark.sql as pyspark_sql
from packaging import version

# Handles optional Spark Connect imports for pyspark>=3.4 (if available)
CURRENT_PYSPARK_VERSION = version.parse(pyspark.__version__)
PYSPARK_CONNECT_AVAILABLE = CURRENT_PYSPARK_VERSION >= version.parse("3.4")
if PYSPARK_CONNECT_AVAILABLE:
    try:
        from pyspark.sql.connect import dataframe as pyspark_connect
    except ImportError:  # pragma: no cover
        # grpcio-status or other Spark Connect deps not installed
        PYSPARK_CONNECT_AVAILABLE = False  # pragma: no cover


@lru_cache
def register_pyspark_backends(
    check_cls_fqn: str | None = None,
    use_narwhals_backend: bool = False,
):
    """Register backends for PySpark DataFrame types.

    Uses the Narwhals backends when ``use_narwhals_backend`` is ``True`` (from
    ``PANDERA_USE_NARWHALS_BACKEND`` or ``pandera.config.CONFIG``); otherwise
    registers the native PySpark backends.

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
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The Narwhals backend is enabled but the 'narwhals' "
                "package is not installed. Install it with: "
                "pip install 'pandera[narwhals]'"
            ) from exc

        import pandera.backends.narwhals.builtin_checks  # noqa: F401
        from pandera.api.checks import Check
        from pandera.api.pyspark.components import Column
        from pandera.api.pyspark.container import DataFrameSchema
        from pandera.backends.narwhals.checks import NarwhalsCheckBackend
        from pandera.backends.narwhals.components import ColumnBackend
        from pandera.backends.narwhals.container import DataFrameSchemaBackend

        DataFrameSchema.register_backend(
            pyspark_sql.DataFrame, DataFrameSchemaBackend, force=True
        )
        Column.register_backend(
            pyspark_sql.DataFrame, ColumnBackend, force=True
        )
        Check.register_backend(
            pyspark_sql.DataFrame, NarwhalsCheckBackend, force=True
        )
        # nw.DataFrame is intentionally NOT registered: PySpark SQL frames are always
        # SQL-lazy under narwhals (exposed as nw.LazyFrame, never nw.DataFrame). This
        # mirrors pandera/backends/ibis/register.py; contrast with polars, which also
        # registers nw.DataFrame because polars frames can be eager.
        Check.register_backend(nw.LazyFrame, NarwhalsCheckBackend, force=True)

        if PYSPARK_CONNECT_AVAILABLE:
            DataFrameSchema.register_backend(
                pyspark_connect.DataFrame, DataFrameSchemaBackend, force=True
            )
            Column.register_backend(
                pyspark_connect.DataFrame, ColumnBackend, force=True
            )
            Check.register_backend(
                pyspark_connect.DataFrame, NarwhalsCheckBackend, force=True
            )
    else:
        from pandera._patch_numpy2 import _patch_numpy2

        _patch_numpy2()

        from pandera.api.checks import Check
        from pandera.api.dataframe.components import ComponentSchema
        from pandera.api.pyspark.components import Column
        from pandera.api.pyspark.container import DataFrameSchema
        from pandera.backends.pyspark import builtin_checks  # noqa: F401
        from pandera.backends.pyspark.checks import PySparkCheckBackend
        from pandera.backends.pyspark.column import ColumnSchemaBackend
        from pandera.backends.pyspark.components import (
            ColumnBackend as PySparkColumnBackend,
        )
        from pandera.backends.pyspark.container import (
            DataFrameSchemaBackend as PySparkDataFrameSchemaBackend,
        )

        # Register PySpark SQL DataFrame
        Check.register_backend(
            pyspark_sql.DataFrame, PySparkCheckBackend, force=True
        )
        ComponentSchema.register_backend(
            pyspark_sql.DataFrame, ColumnSchemaBackend, force=True
        )
        Column.register_backend(
            pyspark_sql.DataFrame, PySparkColumnBackend, force=True
        )
        DataFrameSchema.register_backend(
            pyspark_sql.DataFrame, PySparkDataFrameSchemaBackend, force=True
        )

        # Note: pyspark.pandas DataFrames use pandas-like API and should use
        # the pandas backends which are registered in pandera.backends.pandas.register

        # Register Spark Connect DataFrame, if available
        if PYSPARK_CONNECT_AVAILABLE:
            Check.register_backend(
                pyspark_connect.DataFrame, PySparkCheckBackend, force=True
            )
            ComponentSchema.register_backend(
                pyspark_connect.DataFrame, ColumnSchemaBackend, force=True
            )
            Column.register_backend(
                pyspark_connect.DataFrame, PySparkColumnBackend, force=True
            )
            DataFrameSchema.register_backend(
                pyspark_connect.DataFrame,
                PySparkDataFrameSchemaBackend,
                force=True,
            )
