"""Koalas test suite setup."""

import os

try:
    # pylint: disable=import-outside-toplevel,unused-import
    import databricks.koalas

    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
except ImportError:
    pass
