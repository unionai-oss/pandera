"""Custom type aliases."""

from typing import Union

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame

PandasObject = Union[pd.Series, pd.DataFrame]
PandasExtensionType = pd.core.dtypes.base.ExtensionDtype
PandasDataType = Union[pd.core.dtypes.base.ExtensionDtype, np.dtype, type]
PysparkObject = Union[DataFrame]