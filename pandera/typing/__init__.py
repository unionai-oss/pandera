"""Typing module.

For backwards compatibility, pandas types are exposed to the top-level scope of
the typing module.
"""

from functools import lru_cache
from typing import Set, Type
from pandera.typing.common import AnnotationInfo

try:
    from pandera.typing.pandas import (
        DataFrame,
        Index,
        Series,
        Bool,
        Category,
        Date,
        DateTime,
        Decimal,
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
        INT8,
        INT16,
        INT32,
        INT64,
        UINT8,
        UINT16,
        UINT32,
        UINT64,
        STRING,
    )
except ImportError:
    pass


@lru_cache
def get_dataframe_types():
    from pandera.typing import (
        dask,
        geopandas,
        modin,
        pyspark,
        pyspark_sql,
    )

    dataframe_types: Set[Type] = {DataFrame}
    if dask.DASK_INSTALLED:
        dataframe_types.update({dask.DataFrame})

    if modin.MODIN_INSTALLED:
        dataframe_types.update({modin.DataFrame})

    if pyspark.PYSPARK_INSTALLED:
        dataframe_types.update({pyspark.DataFrame})

    if pyspark_sql.PYSPARK_SQL_INSTALLED:
        dataframe_types.update({pyspark_sql.DataFrame})

    if geopandas.GEOPANDAS_INSTALLED:
        dataframe_types.update({geopandas.GeoDataFrame})

    return dataframe_types


@lru_cache
def get_series_types():
    from pandera.typing import (
        dask,
        geopandas,
        modin,
        pyspark,
    )

    series_types: Set[Type] = {Series}
    if dask.DASK_INSTALLED:
        series_types.update({dask.Series})

    if modin.MODIN_INSTALLED:
        series_types.update({modin.Series})

    if pyspark.PYSPARK_INSTALLED:
        series_types.update({pyspark.Series})

    if geopandas.GEOPANDAS_INSTALLED:
        series_types.update({geopandas.GeoSeries})

    return series_types


@lru_cache
def get_index_types():
    from pandera.typing import dask, modin, pyspark

    index_types: Set[Type] = {Index}
    if dask.DASK_INSTALLED:
        index_types.update({dask.Index})

    if modin.MODIN_INSTALLED:
        index_types.update({modin.Index})

    if pyspark.PYSPARK_INSTALLED:
        index_types.update({pyspark.Index})  # type: ignore [arg-type]

    return index_types


__all__ = [
    "AnnotationInfo",
    "DataFrame",
    "Series",
    "Index",
    "get_dataframe_types",
    "get_index_types",
    "get_series_types",
]
