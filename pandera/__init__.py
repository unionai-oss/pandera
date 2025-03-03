# pylint: disable=wrong-import-position
"""A flexible and expressive dataframe validation library."""

import sys
import warnings
from packaging.version import parse


try:
    import pandas as pd
except ImportError:
    _pandas_installed = False
else:
    _pandas_installed = True
    import numpy as np


if not _pandas_installed:
    sys.exit(0)


_min_pandas_version = parse("2.1.1")
_min_numpy_version = parse("1.24.4")


if parse(pd.__version__) < _min_pandas_version:
    raise ImportError(
        "pandera requires pandas >= 2.1.1, but you have pandas "
        f"{pd.__version__}. Please upgrade pandas to the minimum supported version."
    )

if parse(np.__version__) < _min_numpy_version:
    raise ImportError(
        "pandera requires numpy >= 1.24.4, but you have numpy "
        f"{np.__version__}. Please upgrade numpy to the minimum supported version."
    )


warnings.warn(
    "The pandera module for pandas data validation has been moved to "
    "`pandera.pandas`. Please update your import statement to "
    "`import pandera.pandas as pa`.",
    FutureWarning,
)

from pandera.pandas import *
