import pandas as pd

from pandera.backends.pandas.checks import PandasCheckBackend

from .checks import Check
from .pandas import checks
from .pandas.array import SeriesSchema
from .pandas.components import Column, Index, MultiIndex
from .pandas.container import DataFrameSchema

Check.register_backend(pd.DataFrame, PandasCheckBackend)
Check.register_backend(pd.Series, PandasCheckBackend)
