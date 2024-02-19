"""Custom type aliases."""

from typing import Union

import numpy as np
import pandas as pd

try:
    from pyspark.sql import DataFrame

    PYSPARK_INSTALLED = True
except ImportError:  # pragma: no cover
    PYSPARK_INSTALLED = False

try:
    import polars as pl

    POLARS_INSTALLED = True
except ImportError:  # pragma: no cover
    POLARS_INSTALLED = False

PandasObject = Union[pd.Series, pd.DataFrame]
PandasExtensionType = pd.core.dtypes.base.ExtensionDtype
PandasDataType = Union[pd.core.dtypes.base.ExtensionDtype, np.dtype, type]

if PYSPARK_INSTALLED:
    PysparkObject = Union[DataFrame]

if POLARS_INSTALLED:
    PolarsObject = Union[pl.Series, pl.DataFrame]
