"""Register pyspark backends."""

from functools import lru_cache
from typing import Optional
from packaging import version

import pyspark
import pyspark.sql as ps

# Handles optional Spark Connect imports for pyspark>=3.4 (if available)
CURRENT_PYSPARK_VERSION = version.parse(pyspark.__version__)
if CURRENT_PYSPARK_VERSION >= version.parse("3.4"):
    from pyspark.sql.connect import dataframe as psc


@lru_cache
def register_pyspark_backends(
    check_cls_fqn: Optional[str] = None,
):  # pylint: disable=unused-argument
    """Register pyspark backends.

    This function is called at schema initialization in the _register_*_backends
    method.
    """

    # pylint: disable=import-outside-toplevel,unused-import,cyclic-import
    from pandera._patch_numpy2 import _patch_numpy2

    _patch_numpy2()

    from pandera.api.checks import Check
    from pandera.api.pyspark.column_schema import ColumnSchema
    from pandera.api.pyspark.components import Column
    from pandera.api.pyspark.container import DataFrameSchema
    from pandera.backends.pyspark import builtin_checks
    from pandera.backends.pyspark.checks import PySparkCheckBackend
    from pandera.backends.pyspark.column import ColumnSchemaBackend
    from pandera.backends.pyspark.components import ColumnBackend
    from pandera.backends.pyspark.container import DataFrameSchemaBackend

    # Register classical DataFrame
    Check.register_backend(ps.DataFrame, PySparkCheckBackend)
    ColumnSchema.register_backend(ps.DataFrame, ColumnSchemaBackend)
    Column.register_backend(ps.DataFrame, ColumnBackend)
    DataFrameSchema.register_backend(ps.DataFrame, DataFrameSchemaBackend)
    # Register Spark Connect DataFrame, if available
    if CURRENT_PYSPARK_VERSION >= version.parse("3.4"):
        Check.register_backend(psc.DataFrame, PySparkCheckBackend)
        ColumnSchema.register_backend(psc.DataFrame, ColumnSchemaBackend)
        Column.register_backend(psc.DataFrame, ColumnBackend)
        DataFrameSchema.register_backend(psc.DataFrame, DataFrameSchemaBackend)
