"""Configuration for external packages."""

import os

try:
    if os.getenv("SPARK_LOCAL_IP") is None:
        os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
except ImportError:
    pass
