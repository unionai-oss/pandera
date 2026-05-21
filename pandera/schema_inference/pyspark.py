"""Infer PySpark SQL :class:`~pandera.api.pyspark.container.DataFrameSchema` from data."""

from __future__ import annotations

from typing import Any

from pandera.api.pyspark.components import Column
from pandera.api.pyspark.container import DataFrameSchema
from pandera.schema_statistics.pandas import parse_check_statistics
from pandera.schema_statistics.pyspark import (
    infer_pyspark_dataframe_statistics,
)


def infer_dataframe_schema(df: Any) -> DataFrameSchema:
    """Infer a :class:`DataFrameSchema` from a PySpark DataFrame.

    .. note::

        This runs Spark actions (for example ``count`` and per-column
        aggregations) over the full input.

    :param df: Spark DataFrame to infer from.
    :returns: Inferred schema with ``coerce=True``.
    """
    from pyspark.sql import DataFrame as SparkDataFrame

    if not isinstance(df, SparkDataFrame):
        raise TypeError(
            f"Expected pyspark.sql.DataFrame, got {type(df).__name__}"
        )

    stats = infer_pyspark_dataframe_statistics(df)
    columns_stats = stats["columns"]
    if not columns_stats:
        return DataFrameSchema({}, coerce=True)
    return DataFrameSchema(
        columns={
            col: Column(
                dtype=props["dtype"],
                checks=parse_check_statistics(props["checks"]),
                nullable=props["nullable"],
            )
            for col, props in columns_stats.items()
        },
        coerce=True,
    )


def infer_schema(df: Any) -> DataFrameSchema:
    """Infer a PySpark :class:`DataFrameSchema` from a DataFrame.

    :param df: Spark DataFrame to infer from.
    :returns: Inferred schema.
    :raises TypeError: if ``df`` is not a :class:`pyspark.sql.DataFrame`.
    """
    return infer_dataframe_schema(df)
