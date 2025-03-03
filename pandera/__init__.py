# pylint: disable=wrong-import-position
"""A flexible and expressive dataframe validation library."""

import sys

from pandera._version import __version__

try:
    import pandas as pd
except ImportError:
    _pandas_installed = False
else:
    _pandas_installed = True


if not _pandas_installed:
    sys.exit(0)


from pandera.pandas import *
from pandera.pandas import __all__

__all__.append("__version__")
