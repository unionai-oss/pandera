# pylint: disable=wrong-import-position
"""A flexible and expressive dataframe validation library."""

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from pandera._version import __version__


console = Console()


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
    if "pandas" in str(err) or "numpy" in str(err):
        console.print(
            Panel(
                Markdown(_warning_msg),
                title="PandasImportWarning",
                border_style="yellow",
            )
        )
        console.print_exception()
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
