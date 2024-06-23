"""Register pandas backends."""

from functools import lru_cache
from typing import NamedTuple

import pandas as pd

from pandera.backends.pandas.array import SeriesSchemaBackend
from pandera.backends.pandas.checks import PandasCheckBackend
from pandera.backends.pandas.components import (
    ColumnBackend,
    IndexBackend,
    MultiIndexBackend,
)
from pandera.backends.pandas.container import DataFrameSchemaBackend
from pandera.backends.pandas.hypotheses import PandasHypothesisBackend
from pandera.backends.pandas.parsers import PandasParserBackend


class BackendTypes(NamedTuple):

    # list of datatypes available
    dataframe_datatypes: list
    series_datatypes: list
    index_datatypes: list
    multiindex_datatypes: list
    check_backend_types: list

    # installed backend flags
    dask_installed: bool
    modin_installed: bool
    pyspark_installed: bool
    geopandas_installed: bool


@lru_cache
def get_backend_types():

    import pandera.typing.dask
    import pandera.typing.modin
    import pandera.typing.pyspark
    import pandera.typing.geopandas

    dataframe_datatypes = [pd.DataFrame]
    series_datatypes = [pd.Series]
    index_datatypes = [pd.Index]
    multiindex_datatypes = [pd.MultiIndex]

    if pandera.typing.dask.DASK_INSTALLED:
        import dask.dataframe as dd

        dataframe_datatypes.append(dd.DataFrame)
        series_datatypes.append(dd.Series)
        index_datatypes.append(dd.Index)

    if pandera.typing.modin.MODIN_INSTALLED:
        import modin.pandas as mpd

        dataframe_datatypes.append(mpd.DataFrame)
        series_datatypes.append(mpd.Series)
        index_datatypes.append(mpd.Index)
        multiindex_datatypes.append(mpd.MultiIndex)

    if pandera.typing.pyspark.PYSPARK_INSTALLED:
        import pyspark.pandas as ps

        dataframe_datatypes.append(ps.DataFrame)
        series_datatypes.append(ps.Series)
        index_datatypes.append(ps.Index)
        multiindex_datatypes.append(ps.MultiIndex)

    if pandera.typing.geopandas.GEOPANDAS_INSTALLED:
        import geopandas as gpd

        dataframe_datatypes.append(gpd.GeoDataFrame)
        series_datatypes.append(gpd.GeoSeries)

    check_backend_types = [
        *dataframe_datatypes,
        *series_datatypes,
        *index_datatypes,
    ]

    return BackendTypes(
        dataframe_datatypes=dataframe_datatypes,
        series_datatypes=series_datatypes,
        index_datatypes=index_datatypes,
        multiindex_datatypes=multiindex_datatypes,
        check_backend_types=check_backend_types,
        dask_installed=pandera.typing.dask.DASK_INSTALLED,
        modin_installed=pandera.typing.modin.MODIN_INSTALLED,
        pyspark_installed=pandera.typing.pyspark.PYSPARK_INSTALLED,
        geopandas_installed=pandera.typing.geopandas.GEOPANDAS_INSTALLED,
    )


def register_pandas_backends():
    """Register pandas backends.

    This function is called at schema initialization in the _register_*_backends
    method.
    """

    # pylint: disable=import-outside-toplevel,unused-import,cyclic-import
    from pandera._patch_numpy2 import _patch_numpy2

    _patch_numpy2()

    from pandera.api.checks import Check
    from pandera.api.hypotheses import Hypothesis
    from pandera.api.pandas.array import SeriesSchema
    from pandera.api.pandas.components import Column, Index, MultiIndex
    from pandera.api.pandas.container import DataFrameSchema
    from pandera.api.parsers import Parser
    from pandera.backends.pandas import builtin_checks, builtin_hypotheses

    backend_types = get_backend_types()

    for t in backend_types.check_backend_types:
        Check.register_backend(t, PandasCheckBackend)
        Hypothesis.register_backend(t, PandasHypothesisBackend)
        Parser.register_backend(t, PandasParserBackend)

    for t in backend_types.dataframe_datatypes:
        DataFrameSchema.register_backend(t, DataFrameSchemaBackend)
        Column.register_backend(t, ColumnBackend)
        MultiIndex.register_backend(t, MultiIndexBackend)
        Index.register_backend(t, IndexBackend)

    for t in backend_types.series_datatypes:
        SeriesSchema.register_backend(t, SeriesSchemaBackend)
        Column.register_backend(t, ColumnBackend)
        MultiIndex.register_backend(t, MultiIndexBackend)
        Index.register_backend(t, IndexBackend)

    for t in backend_types.index_datatypes:
        Index.register_backend(t, IndexBackend)

    for t in backend_types.multiindex_datatypes:
        MultiIndex.register_backend(t, MultiIndexBackend)

    if backend_types.dask_installed:
        from pandera.accessors import dask_accessor

    if backend_types.pyspark_installed:
        from pandera.accessors import pyspark_accessor

    if backend_types.modin_installed:
        from pandera.accessors import modin_accessor
