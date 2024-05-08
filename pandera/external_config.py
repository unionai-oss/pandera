"""Configuration for external packages."""

import os

is_spark_local_ip_dirty = False
is_pyarrow_ignore_timezone_dirty = False

try:
    # try importing pyspark to see if it exists. This is important because the
    # pandera.typing module defines a Series type that inherits from
    # pandas.Series, and pyspark v1+ injects a __getitem__ method to pandas
    # Series and DataFrames to support type hinting:
    # https://spark.apache.org/docs/3.2.0/api/python/user_guide/pandas_on_spark/typehints.html#type-hinting-with-names
    # pylint: disable=unused-import
    if os.getenv("SPARK_LOCAL_IP") is None:
        is_spark_local_ip_dirty = True
        os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
    if os.getenv("PYARROW_IGNORE_TIMEZONE") is None:
        is_pyarrow_ignore_timezone_dirty = True
        # This can be overriden by the user
        os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

    import pyspark.pandas
except (ImportError, ModuleNotFoundError):
    pass
finally:
    if is_spark_local_ip_dirty:
        os.environ.pop("SPARK_LOCAL_IP")
    if is_pyarrow_ignore_timezone_dirty:
        os.environ.pop("PYARROW_IGNORE_TIMEZONE")
