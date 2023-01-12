"""Pandera core module.

This module contains the schema specifications for all supported data objects.
"""

import pandas as pd

import pandera.typing
from pandera.backends.pandas.checks import PandasCheckBackend

from .checks import Check
from .pandas import checks
from .pandas.array import SeriesSchema
from .pandas.components import Column, Index, MultiIndex
from .pandas.container import DataFrameSchema

Check.register_backend(pd.DataFrame, PandasCheckBackend)
Check.register_backend(pd.Series, PandasCheckBackend)

if pandera.typing.dask.DASK_INSTALLED:
    import dask.dataframe as dd

    Check.register_backend(dd.DataFrame, PandasCheckBackend)
    Check.register_backend(dd.Series, PandasCheckBackend)

if pandera.typing.modin.MODIN_INSTALLED:
    import modin.pandas as mpd

    Check.register_backend(mpd.DataFrame, PandasCheckBackend)
    Check.register_backend(mpd.Series, PandasCheckBackend)

if pandera.typing.pyspark.PYSPARK_INSTALLED:
    import pyspark.pandas as ps

    Check.register_backend(ps.DataFrame, PandasCheckBackend)
    Check.register_backend(ps.Series, PandasCheckBackend)

if pandera.typing.geopandas.GEOPANDAS_INSTALLED:
    import geopandas as gpd

    Check.register_backend(gpd.GeoDataFrame, PandasCheckBackend)
    Check.register_backend(gpd.GeoSeries, PandasCheckBackend)
