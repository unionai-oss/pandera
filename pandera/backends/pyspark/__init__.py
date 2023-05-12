"""PySpark native backend implementation for schemas and checks."""

import pyspark.sql as pst

from pandera.api.checks import Check
from pandera.backends.pyspark import builtin_checks
from pandera.backends.pyspark.checks import PySparkCheckBackend

for t in [pst.DataFrame]:
    Check.register_backend(t, PySparkCheckBackend)
