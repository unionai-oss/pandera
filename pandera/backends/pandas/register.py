"""Register pandas backends."""

from functools import lru_cache
from typing import NamedTuple, List

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
from pandera.errors import BackendNotFoundError


class BackendTypes(NamedTuple):

    # list of datatypes available
    dataframe_datatypes: list
    series_datatypes: list
    index_datatypes: list
    multiindex_datatypes: list
    check_backend_types: list


PANDAS_LIKE_CLS_NAMES = frozenset(
    [
        "DataFrame",
        "Series",
        "Index",
        "MultiIndex",
        "GeoDataFrame",
        "GeoSeries",
    ]
)


@lru_cache
def get_backend_types(check_cls_fqn: str):

    dataframe_datatypes = []
    series_datatypes = []
    index_datatypes = []
    multiindex_datatypes = []

    mod_name, *mod_path, cls_name = check_cls_fqn.split(".")
    if mod_name != "pandera":
        if cls_name not in PANDAS_LIKE_CLS_NAMES:
            raise BackendNotFoundError(
                f"cls_name {cls_name} not in {PANDAS_LIKE_CLS_NAMES}"
            )

    if mod_name == "pandera":
        # assume mod_path e.g. ["typing", "pandas"]
        assert mod_path[0] == "typing"
        _, mod_name = mod_path

    def register_pandas_backend():
        import pandas as pd
        from pandera.accessors import pandas_accessor

        dataframe_datatypes.append(pd.DataFrame)
        series_datatypes.append(pd.Series)
        index_datatypes.append(pd.Index)
        multiindex_datatypes.append(pd.MultiIndex)

    def register_dask_backend():
        import dask.dataframe as dd
        from pandera.accessors import dask_accessor

        dataframe_datatypes.append(dd.DataFrame)
        series_datatypes.append(dd.Series)
        index_datatypes.append(dd.Index)

    def register_modin_backend():
        import modin.pandas as mpd
        from pandera.accessors import modin_accessor

        dataframe_datatypes.append(mpd.DataFrame)
        series_datatypes.append(mpd.Series)
        index_datatypes.append(mpd.Index)
        multiindex_datatypes.append(mpd.MultiIndex)

    def register_pyspark_backend():
        import pyspark.pandas as ps
        from pandera.accessors import pyspark_accessor

        dataframe_datatypes.append(ps.DataFrame)
        series_datatypes.append(ps.Series)
        index_datatypes.append(ps.Index)
        multiindex_datatypes.append(ps.MultiIndex)

    def register_geopandas_backend():
        import geopandas as gpd

        dataframe_datatypes.append(gpd.GeoDataFrame)
        series_datatypes.append(gpd.GeoSeries)

    {
        "pandas": register_pandas_backend,
        "dask": register_dask_backend,
        "modin": register_modin_backend,
        "pyspark": register_pyspark_backend,
        "geopandas": register_geopandas_backend,
        "pandera": lambda: None,
    }[mod_name]()

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
    )


@lru_cache
def register_pandas_backends(check_cls_fqn: str):
    """Register pandas backends.

    This function is called at schema initialization in the _register_*_backends
    method.

    :param framework_name: name of the framework to register backends for.
        Allowable types are "pandas", "dask", "modin", "pyspark", and
        "geopandas".
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

    backend_types = get_backend_types(check_cls_fqn)

    from pandera.backends.pandas import builtin_checks, builtin_hypotheses

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
