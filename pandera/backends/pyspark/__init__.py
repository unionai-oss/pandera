"""PySpark native backend implementation for schemas and checks."""

from pandera.api.checks import Check

from pandera.backends.pyspark.checks import PySparkCheckBackend
from pandera.backends.pyspark import builtin_checks
import pyspark.sql as pst

for t in [pst.DataFrame]:
    Check.register_backend(t, PySparkCheckBackend)
