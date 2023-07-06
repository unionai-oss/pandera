"""PySpark native backend implementation for schemas and checks."""

import pyspark.sql as pst

from pandera.api.checks import Check
from pandera.api.pyspark.column_schema import ColumnSchema
from pandera.api.pyspark.components import Column
from pandera.api.pyspark.container import DataFrameSchema
from pandera.backends.pyspark import builtin_checks
from pandera.backends.pyspark.checks import PySparkCheckBackend
from pandera.backends.pyspark.column import ColumnSchemaBackend
from pandera.backends.pyspark.components import ColumnBackend
from pandera.backends.pyspark.container import DataFrameSchemaBackend


for t in [pst.DataFrame]:
    Check.register_backend(t, PySparkCheckBackend)
    ColumnSchema.register_backend(t, ColumnSchemaBackend)
    Column.register_backend(t, ColumnBackend)
    DataFrameSchema.register_backend(t, DataFrameSchemaBackend)
