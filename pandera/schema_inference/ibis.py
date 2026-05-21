"""Infer Ibis :class:`~pandera.api.ibis.container.DataFrameSchema` from data."""

from __future__ import annotations

import ibis

from pandera.api.ibis.components import Column
from pandera.api.ibis.container import DataFrameSchema
from pandera.schema_statistics.ibis import infer_ibis_table_statistics
from pandera.schema_statistics.pandas import parse_check_statistics


def infer_dataframe_schema(table: ibis.Table) -> DataFrameSchema:
    """Infer a :class:`DataFrameSchema` from an Ibis table (executes ``table``).

    :param table: Ibis table to infer from.
    :returns: Inferred schema with ``coerce=True``.
    """
    stats = infer_ibis_table_statistics(table)
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


def infer_schema(table: ibis.Table) -> DataFrameSchema:
    """Infer an Ibis :class:`DataFrameSchema` from a table.

    This calls :meth:`ibis.Table.execute` to materialize data for statistics.

    :param table: Ibis table to infer from.
    :returns: Inferred schema.
    :raises TypeError: if ``table`` is not an :class:`ibis.Table`.
    """
    if not isinstance(table, ibis.Table):
        raise TypeError(f"Expected ibis.Table, got {type(table).__name__}")
    return infer_dataframe_schema(table)
