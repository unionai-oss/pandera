# pylint: disable=wrong-import-position
"""A flexible and expressive dataframe validation library."""

from logging import getLogger
import warnings
from pandera._version import __version__


log = getLogger(__name__)


try:
    # Only add pandas to the pandera namespace
    # if pandas and numpy are installed
    import pandas as pd
    import numpy as np

    from pandera.pandas import *
    from pandera.pandas import __all__ as __all_pandas

    __all__ = [
        "__version__",
        *__all_pandas,
    ]

except ImportError as err:
    if "pandas" in str(err) or "numpy" in str(err):
        log.warning(
            f"Pandas and Numpy are required for this version of pandera. {err}",
        )
    else:
        raise  # Re-raise any other `ImportError` exceptions

    from pandera.api.checks import Check
    from pandera.api.dataframe.model_components import (
        Field,
        check,
        dataframe_check,
        dataframe_parser,
        parser,
    )

    __all__ = [
        "__version__",
        "Check",
        "Field",
        "check",
        "dataframe_check",
        "dataframe_parser",
        "parser",
    ]
