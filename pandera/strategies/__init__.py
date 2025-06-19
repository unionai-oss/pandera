# pylint: disable=unused-import
"""Data synthesis strategies for pandera, powered by the hypothesis package."""

import warnings

try:
    import pandas
    from pandera.strategies.pandas_strategies import *
except ImportError:
    pass
