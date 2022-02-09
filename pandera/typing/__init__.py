"""Typing module.

For backwards compatibility, pandas types are exposed to the top-level scope of
the typing module.
"""

from . import dask, fastapi, geopandas, koalas, modin
from .common import (
    BOOL,
    INT8,
    INT16,
    INT32,
    INT64,
    STRING,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
    AnnotationInfo,
    Bool,
    Category,
    DateTime,
    Float,
    Float16,
    Float32,
    Float64,
    Int,
    Int8,
    Int16,
    Int32,
    Int64,
    Object,
    String,
    Timedelta,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
)
from .pandas import DataFrame, Index, Series

DATAFRAME_TYPES = {DataFrame}
SERIES_TYPES = {Series}
INDEX_TYPES = {Index}

if dask.DASK_INSTALLED:
    DATAFRAME_TYPES.update({dask.DataFrame})
    SERIES_TYPES.update({dask.Series})
    INDEX_TYPES.update({dask.Index})

if modin.MODIN_INSTALLED:
    DATAFRAME_TYPES.update({modin.DataFrame})
    SERIES_TYPES.update({modin.Series})
    INDEX_TYPES.update({modin.Index})

if koalas.KOALAS_INSTALLED:
    DATAFRAME_TYPES.update({koalas.DataFrame})
    SERIES_TYPES.update({koalas.Series})
    INDEX_TYPES.update({koalas.Index})


if geopandas.GEOPANDAS_INSTALLED:
    DATAFRAME_TYPES.update({geopandas.GeoDataFrame})
    SERIES_TYPES.update({geopandas.GeoSeries})


__all__ = [
    "DataFrame",
    "Series",
    "Index",
]
