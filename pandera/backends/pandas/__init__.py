"""Pandas backend implementation for schemas and checks."""

import pandas as pd

import pandera.typing
from pandera.api.checks import Check
from pandera.api.hypotheses import Hypothesis

from pandera.backends.pandas.checks import PandasCheckBackend
from pandera.backends.pandas.hypotheses import PandasHypothesisBackend
from pandera.backends.pandas import builtin_checks, builtin_hypotheses
from pandera.backends.pyspark.checks import PySparkCheckBackend
import pyspark.sql as pst


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

for t in [pst.DataFrame]:
    Check.register_backend(t, PySparkCheckBackend)
