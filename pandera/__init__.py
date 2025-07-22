# pylint: disable=wrong-import-position
"""A flexible and expressive dataframe validation library."""

from pandera._version import __version__


_warning_msg = """Pandas and numpy have been removed from the base pandera
dependencies. Please install pandas as part of your environment's
dependencies or install the pandas extra with:

```bash
pip install pandas pandera

# or
pip install 'pandera[pandas]'
```
"""


try:
    # Only add pandas to the top-level pandera namespace
    # if pandas and numpy are installed
    import pandas as pd
    import numpy as np

    from pandera._pandas_deprecated import *
    from pandera._pandas_deprecated import __all__ as _pandas_deprecated_all
    from pandera import dtypes
    from pandera import typing

    __all__ = [
        "__version__",
        *_pandas_deprecated_all,
    ]

except ImportError as err:
    import warnings

    err_msg = str(err)
    if err_msg in {"No module named 'pandas'", "No module named 'numpy'"}:
        warnings.warn(_warning_msg, UserWarning)
    elif err_msg.startswith("pandera requires pandas >= 2.1.1"):
        warnings.warn(err_msg, UserWarning)
    else:
        raise  # Re-raise any other `ImportError` exceptions

    from pandera import dtypes
    from pandera import typing
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
        "dtypes",
        "typing",
    ]
