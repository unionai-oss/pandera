"""Register pyspark backends."""

from functools import lru_cache
from typing import Optional

import pyspark
import pyspark.sql as pyspark_sql
from packaging import version

# Handles optional Spark Connect imports for pyspark>=3.4 (if available)
CURRENT_PYSPARK_VERSION = version.parse(pyspark.__version__)
PYSPARK_CONNECT_AVAILABLE = CURRENT_PYSPARK_VERSION >= version.parse("3.4")
if PYSPARK_CONNECT_AVAILABLE:
    from pyspark.sql.connect import dataframe as pyspark_connect


@lru_cache
def register_pyspark_backends(
    check_cls_fqn: str | None = None,
):
    """Register pyspark backends.

    This function is called at schema initialization in the _register_*_backends
    method.
    """

    from pandera._patch_numpy2 import _patch_numpy2

    _patch_numpy2()

    from pandera.api.checks import Check
    from pandera.api.dataframe.components import ComponentSchema
    from pandera.api.pyspark.components import Column
    from pandera.api.pyspark.container import DataFrameSchema
    from pandera.backends.pyspark import builtin_checks
    from pandera.backends.pyspark.checks import PySparkCheckBackend
    from pandera.backends.pyspark.column import ColumnSchemaBackend
    from pandera.backends.pyspark.components import ColumnBackend
    from pandera.backends.pyspark.container import DataFrameSchemaBackend

    # Register PySpark SQL DataFrame
    Check.register_backend(pyspark_sql.DataFrame, PySparkCheckBackend)
    ComponentSchema.register_backend(
        pyspark_sql.DataFrame, ColumnSchemaBackend
    )
    Column.register_backend(pyspark_sql.DataFrame, ColumnBackend)
    DataFrameSchema.register_backend(
        pyspark_sql.DataFrame, DataFrameSchemaBackend
    )

    # Note: pyspark.pandas DataFrames use pandas-like API and should use
    # the pandas backends which are registered in pandera.backends.pandas.register

    # Register Spark Connect DataFrame, if available
    if PYSPARK_CONNECT_AVAILABLE:
        Check.register_backend(pyspark_connect.DataFrame, PySparkCheckBackend)
        ComponentSchema.register_backend(
            pyspark_connect.DataFrame, ColumnSchemaBackend
        )
        Column.register_backend(pyspark_connect.DataFrame, ColumnBackend)
        DataFrameSchema.register_backend(
            pyspark_connect.DataFrame, DataFrameSchemaBackend
        )
