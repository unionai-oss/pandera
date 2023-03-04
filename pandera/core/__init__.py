"""Pandera core module.

This module contains the schema specifications for all supported data objects.
"""

import pandas as pd

import pandera.typing
from pandera.backends.pandas.checks import PandasCheckBackend
from pandera.backends.pandas.hypotheses import PandasHypothesisBackend

from pandera.core.checks import Check
from pandera.core.hypotheses import Hypothesis
from pandera.core.base import (
    builtin_checks as base_builtin_checks,
    builtin_hypotheses as base_builtin_hypotheses,
)
from pandera.core.pandas import (
    builtin_checks as pandas_builtin_checks,
    builtin_hypotheses as pandas_builtin_hypotheses,
)
from pandera.core.pandas.array import SeriesSchema
from pandera.core.pandas.components import Column, Index, MultiIndex
from pandera.core.pandas.container import DataFrameSchema

data_types = [pd.DataFrame, pd.Series]

if pandera.typing.dask.DASK_INSTALLED:
    import dask.dataframe as dd

    data_types.extend([dd.DataFrame, dd.Series])

if pandera.typing.modin.MODIN_INSTALLED:
    import modin.pandas as mpd

    data_types.extend([mpd.DataFrame, mpd.Series])

if pandera.typing.pyspark.PYSPARK_INSTALLED:
    import pyspark.pandas as ps

    data_types.extend([ps.DataFrame, ps.Series])

if pandera.typing.geopandas.GEOPANDAS_INSTALLED:
    import geopandas as gpd

    data_types.extend([gpd.GeoDataFrame, gpd.GeoSeries])

for t in data_types:
    Check.register_backend(t, PandasCheckBackend)
    Hypothesis.register_backend(t, PandasHypothesisBackend)
