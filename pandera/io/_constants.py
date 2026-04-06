"""Shared constants for :mod:`pandera.io` serialization modules."""

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

MISSING_PYYAML_MESSAGE = (
    "IO and formatting requires 'pyyaml' to be installed.\n"
    "You can install pandera together with the IO dependencies with:\n"
    "pip install pandera[io]\n"
)
