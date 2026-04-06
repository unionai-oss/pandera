"""Statistics extraction for polars :class:`~pandera.api.polars.container.DataFrameSchema`."""

from __future__ import annotations

from typing import Any

from pandera import dtypes
from pandera.engines import polars_engine
from pandera.schema_statistics.pandas import parse_checks


def _infer_polars_series_checks(
    series, data_type: dtypes.DataType
) -> dict[str, Any] | None:
    """Infer check statistics dict for a Polars series (pandas infer parity)."""
    n = len(series)
    if n == 0 or series.null_count() == n:
        return None
    if dtypes.is_datetime(data_type):
        return {
            "greater_than_or_equal_to": series.min(),
            "less_than_or_equal_to": series.max(),
        }
    if dtypes.is_numeric(data_type) and not dtypes.is_bool(data_type):
        return {
            "greater_than_or_equal_to": float(series.min()),
            "less_than_or_equal_to": float(series.max()),
        }
    if dtypes.is_category(data_type):
        return {"isin": series.drop_nulls().unique().to_list()}
    return None


def infer_polars_dataframe_statistics(df: Any) -> dict[str, Any]:
    """Infer column statistics from a Polars :class:`~polars.DataFrame`."""
    import polars as pl

    if not isinstance(df, pl.DataFrame):
        raise TypeError(
            f"Expected a polars DataFrame, got {type(df).__name__}"
        )

    column_statistics: dict[str, Any] = {}
    for name in df.columns:
        series = df[name]
        data_type = polars_engine.Engine.dtype(series.dtype)
        nullable = bool(series.null_count() > 0)
        checks = _infer_polars_series_checks(series, data_type)
        column_statistics[name] = {
            "dtype": data_type,
            "nullable": nullable,
            "checks": checks,
        }

    return {
        "columns": column_statistics if column_statistics else None,
        "index": None,
        "checks": None,
        "coerce": True,
    }


def get_dataframe_schema_statistics(dataframe_schema) -> dict[str, Any]:
    """Get statistical properties from a polars dataframe schema."""
    statistics = {
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
    return statistics
