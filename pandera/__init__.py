# pylint: disable=wrong-import-position
"""A flexible and expressive dataframe validation library."""

import sys


try:
    import pandas as pd
except ImportError:
    _pandas_installed = False
else:
    _pandas_installed = True
    import numpy as np


if not _pandas_installed:
    sys.exit(0)


from pandera.pandas import *
