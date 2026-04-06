"""Statistics extraction for PySpark SQL dataframe schemas."""

from __future__ import annotations

from typing import Any

from pandera.schema_statistics.pandas import parse_checks


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
