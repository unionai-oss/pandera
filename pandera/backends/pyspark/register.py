"""Register pyspark backends."""

import pyspark.sql as pst


def register_pyspark_backends():
    """Register pyspark backends.

    This function is called at schema initialization in the _register_*_backends
    method.
    """

    # pylint: disable=import-outside-toplevel,unused-import,cyclic-import
    from pandera.api.checks import Check
    from pandera.api.pyspark.column_schema import ColumnSchema
    from pandera.api.pyspark.components import Column
    from pandera.api.pyspark.container import DataFrameSchema
    from pandera.backends.pyspark import builtin_checks
    from pandera.backends.pyspark.checks import PySparkCheckBackend
    from pandera.backends.pyspark.column import ColumnSchemaBackend
    from pandera.backends.pyspark.components import ColumnBackend
    from pandera.backends.pyspark.container import DataFrameSchemaBackend

    Check.register_backend(pst.DataFrame, PySparkCheckBackend)
    ColumnSchema.register_backend(pst.DataFrame, ColumnSchemaBackend)
    Column.register_backend(pst.DataFrame, ColumnBackend)
    DataFrameSchema.register_backend(pst.DataFrame, DataFrameSchemaBackend)
