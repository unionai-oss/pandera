"""Infer Polars :class:`~pandera.api.polars.container.DataFrameSchema` from data."""

from __future__ import annotations

from typing import overload

import polars as pl

from pandera.api.polars.components import Column
from pandera.api.polars.container import DataFrameSchema
from pandera.schema_statistics.polars import (
    infer_polars_dataframe_statistics,
    parse_check_statistics,
)


def infer_dataframe_schema(df: pl.DataFrame) -> DataFrameSchema:
    """Infer a :class:`DataFrameSchema` from a Polars DataFrame.

    :param df: Polars DataFrame to infer from.
    :returns: Inferred schema with ``coerce=True``.
    """
    stats = infer_polars_dataframe_statistics(df)
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


@overload
def infer_schema(df: pl.DataFrame) -> DataFrameSchema: ...


@overload
def infer_schema(df: pl.LazyFrame) -> DataFrameSchema: ...


def infer_schema(df: pl.DataFrame | pl.LazyFrame) -> DataFrameSchema:
    """Infer a Polars :class:`DataFrameSchema` from a DataFrame or LazyFrame.

    LazyFrames are collected in memory before inference.

    :param df: Polars DataFrame or LazyFrame.
    :returns: Inferred schema.
    :raises TypeError: if ``df`` is not a Polars DataFrame or LazyFrame.
    """
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    if not isinstance(df, pl.DataFrame):
        raise TypeError(
            "Expected polars.DataFrame or polars.LazyFrame, "
            f"got {type(df).__name__}"
        )
    return infer_dataframe_schema(df)
