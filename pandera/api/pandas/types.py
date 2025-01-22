# pylint: disable=unused-import
"""Utility functions for pandas validation."""

from functools import lru_cache
from typing import Any, NamedTuple, Type, TypeVar, Union, Optional

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


@lru_cache
def get_backend_types(check_cls_fqn: str) -> BackendTypes:

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
    else:
        mod_name = mod_name.split(".")[-1]

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
        "dask": register_dask_backend,
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


T = TypeVar("T")


def _get_fullname(_cls: Type) -> str:
    return f"{_cls.__module__}.{_cls.__name__}"


def get_backend_types_from_mro(_cls: Type) -> Optional[BackendTypes]:
    try:
        return get_backend_types(_get_fullname(_cls))
    except BackendNotFoundError:
        for base_cls in _cls.__bases__:
            try:
                return get_backend_types(_get_fullname(base_cls))
            except BackendNotFoundError:
                pass
        return None


def is_table(obj):
    """Verifies whether an object is table-like.

    Where a table is a 2-dimensional data matrix of rows and columns, which
    can be indexed in multiple different ways.
    """
    backend_types = get_backend_types_from_mro(type(obj))
    return backend_types is not None and isinstance(
        obj, backend_types.dataframe_datatypes
    )


def is_field(obj):
    """Verifies whether an object is field-like.

    Where a field is a columnar representation of data in a table-like
    data structure.
    """
    backend_types = get_backend_types_from_mro(type(obj))
    return backend_types is not None and isinstance(
        obj, backend_types.series_datatypes
    )


def is_index(obj):
    """Verifies whether an object is a table index."""
    backend_types = get_backend_types_from_mro(type(obj))
    return backend_types is not None and isinstance(
        obj, backend_types.index_datatypes
    )


def is_multiindex(obj):
    """Verifies whether an object is a multi-level table index."""
    backend_types = get_backend_types_from_mro(type(obj))
    return backend_types is not None and isinstance(
        obj, backend_types.multiindex_datatypes
    )


def is_table_or_field(obj):
    """Verifies whether an object is table- or field-like."""
    return is_table(obj) or is_field(obj)


def is_bool(x):
    """Verifies whether an object is a boolean type."""
    return isinstance(x, (bool, np.bool_))
