"""Infer :class:`~pandera.api.geopandas.container.GeoDataFrameSchema` from data."""

from __future__ import annotations

from typing import overload

import pandas as pd

from pandera.api.geopandas.container import GeoDataFrameSchema
from pandera.api.pandas.array import SeriesSchema
from pandera.schema_inference.pandas import (
    infer_dataframe_schema as _infer_pandas_df_schema,
)
from pandera.schema_inference.pandas import (
    infer_series_schema,
)


def infer_geodataframe_schema(df: pd.DataFrame) -> GeoDataFrameSchema:
    """Infer a :class:`GeoDataFrameSchema` from a GeoDataFrame.

    :param df: GeoDataFrame to infer from.
    :returns: Inferred :class:`GeoDataFrameSchema` with ``coerce=True``.
    """
    schema = _infer_pandas_df_schema(df)
    return GeoDataFrameSchema._from_dataframe_schema(schema)


@overload
def infer_schema(
    pandas_obj: pd.DataFrame,
) -> GeoDataFrameSchema: ...


@overload
def infer_schema(  # type: ignore[overload-cannot-match]
    pandas_obj: pd.Series,
) -> SeriesSchema: ...


def infer_schema(pandas_obj):
    """Infer schema for a GeoDataFrame, DataFrame, or Series.

    When *pandas_obj* is a :class:`geopandas.GeoDataFrame` or
    :class:`pandas.DataFrame` this returns a :class:`GeoDataFrameSchema`.
    For a :class:`pandas.Series` it returns a :class:`SeriesSchema`.

    :param pandas_obj: GeoDataFrame, DataFrame, or Series to infer.
    :returns: GeoDataFrameSchema or SeriesSchema.
    :raises TypeError: if *pandas_obj* is not a recognised type.
    """
    if isinstance(pandas_obj, pd.DataFrame):
        return infer_geodataframe_schema(pandas_obj)
    elif isinstance(pandas_obj, pd.Series):
        return infer_series_schema(pandas_obj)
    else:
        raise TypeError(
            "pandas_obj type not recognized. Expected a "
            "GeoDataFrame, DataFrame, or Series, "
            f"found {type(pandas_obj)}"
        )
