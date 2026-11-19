"""Configuration for external packages."""


def _set_pyspark_environment_variables():
    """Import pyspark.pandas, if available, without touching os.environ.

    Pandera intentionally does not set any pyspark-related environment
    variables (e.g. ``SPARK_LOCAL_IP`` or ``PYARROW_IGNORE_TIMEZONE``):
    doing so overrides Spark's default behavior in the caller's process and
    can break distributed setups. Users who need them should set them
    before importing pandera. See the pyspark docs for details.
    """

    try:
        # try importing pyspark to see if it exists. This is important because the
        # pandera.typing module defines a Series type that inherits from
        # pandas.Series, and pyspark v1+ injects a __getitem__ method to pandas
        # Series and DataFrames to support type hinting:
        # https://spark.apache.org/docs/3.2.0/api/python/user_guide/pandas_on_spark/typehints.html#type-hinting-with-names

        from pandera._patch_numpy2 import _patch_numpy2

        _patch_numpy2()

        import pyspark.pandas
    except (ImportError, ModuleNotFoundError):
        pass
