# pylint: disable=unused-import
"""Data synthesis strategies for pandera, powered by the hypothesis package."""

import warnings

try:
    import pandas
    from pandera.strategies.pandas_strategies import *

    warnings.warn(
        "The pandas data synthesis strategies have been moved to "
        "`pandera.strategies.pandas_strategies`. Please update your import "
        "statement to `from pandera.strategies import pandas_strategies as st`.",
        FutureWarning,
    )
except ImportError:
    pass
