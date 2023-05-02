"""Module for inferring dataframe/series schema."""

from typing import overload

import pandas as pd

from pandera.api.pandas.array import SeriesSchema
from pandera.api.pandas.components import Column, Index, MultiIndex
from pandera.api.pandas.container import DataFrameSchema
from pandera.schema_statistics.pandas import (
    infer_dataframe_statistics,
    infer_series_statistics,
    parse_check_statistics,
)


@overload
def infer_schema(
    pandas_obj: pd.Series,
) -> SeriesSchema:  # pragma: no cover
    ...


@overload
def infer_schema(  # type: ignore[misc]
    pandas_obj: pd.DataFrame,
) -> DataFrameSchema:  # pragma: no cover
    ...


def infer_schema(pandas_obj):
    """Infer schema for pandas DataFrame or Series object.

    :param pandas_obj: DataFrame or Series object to infer.
    :returns: DataFrameSchema or SeriesSchema
    :raises: TypeError if pandas_obj is not expected type.
    """
    if isinstance(pandas_obj, pd.DataFrame):
        return infer_dataframe_schema(pandas_obj)
    elif isinstance(pandas_obj, pd.Series):
        return infer_series_schema(pandas_obj)
    else:
        raise TypeError(
            "pandas_obj type not recognized. Expected a pandas DataFrame or "
            f"Series, found {type(pandas_obj)}"
        )


def _create_index(index_statistics):
    index = [
        Index(
            properties["dtype"],
            checks=parse_check_statistics(properties["checks"]),
            nullable=properties["nullable"],
            name=properties["name"],
        )
        for properties in index_statistics
    ]
    if len(index) == 1:
        index = index[0]  # type: ignore
    else:
        index = MultiIndex(index)  # type: ignore

    return index


def infer_dataframe_schema(df: pd.DataFrame) -> DataFrameSchema:
    """Infer a DataFrameSchema from a pandas DataFrame.

    :param df: DataFrame object to infer.
    :returns: DataFrameSchema
    """
    df_statistics = infer_dataframe_statistics(df)
    schema = DataFrameSchema(
        columns={
            colname: Column(
                properties["dtype"],
                checks=parse_check_statistics(properties["checks"]),
                nullable=properties["nullable"],
            )
            for colname, properties in df_statistics["columns"].items()
        },
        index=_create_index(df_statistics["index"]),
        coerce=True,
    )
    schema._is_inferred = True
    return schema


def infer_series_schema(series) -> SeriesSchema:
    """Infer a SeriesSchema from a pandas DataFrame.

    :param series: Series object to infer.
    :returns: SeriesSchema
    """
    series_statistics = infer_series_statistics(series)
    schema = SeriesSchema(
        dtype=series_statistics["dtype"],
        checks=parse_check_statistics(series_statistics["checks"]),
        nullable=series_statistics["nullable"],
        name=series_statistics["name"],
        coerce=True,
    )
    schema._is_inferred = True
    return schema
