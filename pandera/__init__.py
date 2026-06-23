"""A flexible and expressive dataframe validation library."""

from pandera._version import __version__

try:
    # Only add pandas to the top-level pandera namespace
    # if pandas and numpy are installed
    import numpy as np
    import pandas as pd

    from pandera import dtypes, typing
    from pandera._pandas_deprecated import *
    from pandera._pandas_deprecated import __all__ as _pandas_deprecated_all
    from pandera.config import set_config

    __all__ = [
        "__version__",
        "set_config",
        *_pandas_deprecated_all,
    ]

except (ImportError, ModuleNotFoundError) as err:
    import warnings

    err_msg = str(err)
    if err_msg.startswith("pandera requires pandas >= 2.1.1"):
        warnings.warn(err_msg, UserWarning)
    elif err_msg.startswith(
        ("No module named 'pandas'", "No module named 'numpy'")
    ):
        # ignore this error
        pass
    else:
        raise  # Re-raise any other `ImportError` exceptions

    # Register the builtin check functions (e.g. greater_than_or_equal_to,
    # isin). Without pandas/numpy the pandas import above fails before
    # ``_pandas_deprecated`` (which normally triggers this) is imported, so
    # register them here too. Otherwise builtin checks raise KeyError when
    # constructed in a polars-only install. See GH #2387.
    import pandera.backends.base.builtin_checks  # noqa: F401
    from pandera import dtypes, typing
    from pandera.api.checks import Check
    from pandera.api.dataframe.model_components import (
        Field,
        check,
        dataframe_check,
        dataframe_parser,
        parser,
    )
    from pandera.config import set_config

    __all__ = [
        "__version__",
        "set_config",
        "Check",
        "Field",
        "check",
        "dataframe_check",
        "dataframe_parser",
        "parser",
        "dtypes",
        "typing",
    ]
