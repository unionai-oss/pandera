"""Configuration for external packages."""

import os

try:
    # NOTE: don't rely on this, due to potential performance issues. For more
    # details, see:
    # https://koalas.readthedocs.io/en/latest/user_guide/options.html#operations-on-different-dataframes
    import databricks.koalas as ks

    if os.getenv("SPARK_LOCAL_IP") is None:
        os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
    ks.set_option("compute.ops_on_diff_frames", True)
except ImportError:
    pass
