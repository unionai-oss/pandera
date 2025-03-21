# pylint: disable=wrong-import-position
"""A flexible and expressive dataframe validation library."""


from logging import getLogger

from pandera._version import __version__


log = getLogger(__name__)


try:
    #  Only add pandas to the pandera namespace
    #  if pandas and numpy are installed
    import pandas as pd
    import numpy as np

    from pandera.pandas import *
    from pandera.pandas import __all__
    __all__.append("__version__")

except ImportError as err:
    if 'pandas' or 'numpy' in str(err):
        log.warn(
            'Pandas and Numpy are required for this version of pandera.',
            err,
        )
    else:
        raise  # Re-raise any other `ImportError` exceptions

    __all__ = ["__version__", ]
