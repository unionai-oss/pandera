"""Statistics extraction for Ibis :class:`~pandera.api.ibis.container.DataFrameSchema`."""

from __future__ import annotations

from typing import Any

import pandas as pd

from pandera.engines import ibis_engine
from pandera.schema_statistics.pandas import (
    infer_dataframe_statistics,
    parse_checks,
)


def infer_ibis_table_statistics(table: Any) -> dict[str, Any]:
    """Infer column statistics from an Ibis :class:`~ibis.Table` (executes the table)."""
    import ibis

    if not isinstance(table, ibis.Table):
        raise TypeError(f"Expected an ibis.Table, got {type(table).__name__}")

    pdf = table.execute()
    if not isinstance(pdf, pd.DataFrame):
        pdf = pd.DataFrame(pdf)

    pstats = infer_dataframe_statistics(pdf)
    columns: dict[str, Any] = {}
    for col in table.columns:
        ibis_dt = ibis_engine.Engine.dtype(table[col].type())
        col_stats = pstats["columns"][col]
        columns[col] = {
            "dtype": ibis_dt,
            "nullable": col_stats["nullable"],
            "checks": col_stats["checks"],
        }

    return {
        "columns": columns,
        "index": None,
        "checks": None,
        "coerce": True,
    }


def get_dataframe_schema_statistics(dataframe_schema) -> dict[str, Any]:
    """Get statistical properties from an Ibis table schema."""
    return {
        "columns": {
            col_name: {
                "dtype": column.dtype,
                "nullable": column.nullable,
                "coerce": column.coerce,
                "required": column.required,
                "regex": column.regex,
                "checks": parse_checks(column.checks),
                "unique": column.unique,
                "description": column.description,
                "title": column.title,
                "default": column.default,
                "report_duplicates": column.report_duplicates,
                "drop_invalid_rows": column.drop_invalid_rows,
            }
            for col_name, column in dataframe_schema.columns.items()
        },
        "checks": parse_checks(dataframe_schema.checks),
        "index": None,
        "coerce": dataframe_schema.coerce,
    }
