# pylint: disable=wrong-import-position
"""A flexible and expressive dataframe validation library."""

import sys


try:
    import pandas as pd
except ImportError:
    PANDAS_INSTALLED = False
else:
    PANDAS_INSTALLED = True
    # check minimum pandas version


if not PANDAS_INSTALLED:
    sys.exit(0)


from pandera.pandas import *
