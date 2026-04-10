"""Statistics extraction for PySpark SQL dataframe schemas."""

from __future__ import annotations

from typing import Any

from pandera import dtypes
from pandera.engines import pyspark_engine
from pandera.schema_statistics.pandas import parse_checks


def infer_pyspark_dataframe_statistics(df: Any) -> dict[str, Any]:
    """Infer column statistics from a PySpark :class:`~pyspark.sql.DataFrame`."""
    from pyspark.sql import DataFrame
    from pyspark.sql import functions as F

    if not isinstance(df, DataFrame):
        raise TypeError(
            f"Expected a pyspark.sql.DataFrame, got {type(df).__name__}"
        )

    schema = df.schema
    cols = df.columns
    if not cols:
        return {
            "columns": None,
            "index": None,
            "checks": None,
            "coerce": True,
        }

    nrow = df.count()
    null_aggs = [
        F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(f"__nn_{i}")
        for i, c in enumerate(cols)
    ]
    null_row = df.select(null_aggs).collect()[0]

    column_statistics: dict[str, Any] = {}
    for i, field in enumerate(schema.fields):
        c = field.name
        nulls = int(null_row[f"__nn_{i}"])
        nullable = nrow > 0 and nulls > 0
        pdt = pyspark_engine.Engine.dtype(field.dataType)
        checks = None
        if nrow > 0 and nulls < nrow:
            if dtypes.is_datetime(pdt) or (
                dtypes.is_numeric(pdt) and not dtypes.is_bool(pdt)
            ):
                row = df.select(F.min(c), F.max(c)).collect()[0]
                checks = {
                    "greater_than_or_equal_to": row[0],
                    "less_than_or_equal_to": row[1],
                }
            elif dtypes.is_category(pdt):
                dist = df.select(c).distinct().limit(256).collect()
                checks = {"isin": [r[0] for r in dist]}
        column_statistics[c] = {
            "dtype": pdt,
            "nullable": nullable,
            "checks": checks,
        }

    return {
        "columns": column_statistics,
        "index": None,
        "checks": None,
        "coerce": True,
    }


def get_dataframe_schema_statistics(dataframe_schema) -> dict[str, Any]:
    """Get statistical properties from a PySpark :class:`DataFrameSchema`."""
    return {
        "columns": {
            col_name: {
                "dtype": column.dtype,
                "nullable": column.nullable,
                "coerce": column.coerce,
                "required": column.required,
                "regex": column.regex,
                "checks": parse_checks(column.checks),
                "description": column.description,
                "title": column.title,
            }
            for col_name, column in dataframe_schema.columns.items()
        },
        "checks": parse_checks(dataframe_schema.checks),
        "index": None,
        "coerce": dataframe_schema.coerce,
    }
