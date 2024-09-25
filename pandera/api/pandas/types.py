# pylint: disable=unused-import
"""Utility functions for pandas validation."""

from functools import lru_cache
from typing import Any, NamedTuple, Type, Union

import numpy as np
import pandas as pd

from pandera.dtypes import DataType
from pandera.errors import BackendNotFoundError


PandasDtypeInputTypes = Union[
    str,
    type,
    DataType,
    Type,
    pd.core.dtypes.base.ExtensionDtype,
    np.dtype,
]

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


class BackendTypes(NamedTuple):

    # list of datatypes available
    dataframe_datatypes: tuple
    series_datatypes: tuple
    index_datatypes: tuple
    multiindex_datatypes: tuple
    check_backend_types: tuple


def _get_fullname(obj: Any) -> str:
    _type = type(obj)
    return f"{_type.__module__}.{_type.__name__}"


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
        *_, mod_name = mod_path

    def register_pandas_backend():
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

        register_pandas_backend()
        dataframe_datatypes.append(gpd.GeoDataFrame)
        series_datatypes.append(gpd.GeoSeries)

    register_fn = {
        "pandas": register_pandas_backend,
        "dask_expr": register_dask_backend,
        "modin": register_modin_backend,
        "pyspark": register_pyspark_backend,
        "geopandas": register_geopandas_backend,
        "pandera": lambda: None,
    }[mod_name]

    register_fn()

    check_backend_types = [
        *dataframe_datatypes,
        *series_datatypes,
        *index_datatypes,
    ]

    return BackendTypes(
        dataframe_datatypes=tuple(dataframe_datatypes),
        series_datatypes=tuple(series_datatypes),
        index_datatypes=tuple(index_datatypes),
        multiindex_datatypes=tuple(multiindex_datatypes),
        check_backend_types=tuple(check_backend_types),
    )


def is_table(obj):
    """Verifies whether an object is table-like.

    Where a table is a 2-dimensional data matrix of rows and columns, which
    can be indexed in multiple different ways.
    """
    return isinstance(
        obj, get_backend_types(_get_fullname(obj)).dataframe_datatypes
    )


def is_field(obj):
    """Verifies whether an object is field-like.

    Where a field is a columnar representation of data in a table-like
    data structure.
    """
    return isinstance(
        obj, get_backend_types(_get_fullname(obj)).series_datatypes
    )


def is_index(obj):
    """Verifies whether an object is a table index."""
    return isinstance(
        obj, get_backend_types(_get_fullname(obj)).index_datatypes
    )


def is_multiindex(obj):
    """Verifies whether an object is a multi-level table index."""
    return isinstance(
        obj, get_backend_types(_get_fullname(obj)).multiindex_datatypes
    )


def is_table_or_field(obj):
    """Verifies whether an object is table- or field-like."""
    return is_table(obj) or is_field(obj)


def is_bool(x):
    """Verifies whether an object is a boolean type."""
    return isinstance(x, (bool, np.bool_))
