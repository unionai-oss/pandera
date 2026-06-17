"""Statistics extraction for polars :class:`~pandera.api.polars.container.DataFrameSchema`."""

from __future__ import annotations

import warnings
from typing import Any, Union

from pandera import dtypes
from pandera.api.checks import Check
from pandera.engines import polars_engine


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


def parse_checks(checks) -> Union[list[dict[str, Any]], None]:
    """Convert Check object to check statistics including options."""

    def _has_custom_error(check: Check) -> bool:
        """Determine whether a check has a user-defined error message."""
        if check.error is None:
            return False

        if check.name is None or not Check.is_builtin_check(check.name):
            return True

        try:
            default_check = getattr(Check, check.name)(
                **(check.statistics or {})
            )
        except (AttributeError, TypeError, ValueError):
            return True

        return check.error != default_check.error

    check_statistics = []

    for check in checks:
        if check not in Check:
            warnings.warn(
                "Only registered checks may be serialized to statistics. "
                "Did you forget to register it with the extension API? "
                f"Check `{check.name}` will be skipped."
            )
            continue

        base_stats = {} if check.statistics is None else check.statistics

        check_options = {
            "check_name": check.name,
            "raise_warning": check.raise_warning,
            "n_failure_cases": check.n_failure_cases,
            "ignore_na": check.ignore_na,
        }
        if _has_custom_error(check):
            check_options["error"] = check.error

        check_options = {
            k: v for k, v in check_options.items() if v is not None
        }

        if check_options:
            base_stats["options"] = check_options
            check_statistics.append(base_stats)

    return check_statistics if check_statistics else None


def parse_check_statistics(check_stats: Union[dict[str, Any], None]):
    """Convert check statistics to a list of Check objects, including their options."""
    if check_stats is None:
        return None
    checks = []
    for check_name, stats in check_stats.items():
        check = getattr(Check, check_name)
        try:
            if isinstance(stats, dict):
                options = (
                    stats.pop("options", {}) if "options" in stats else {}
                )
                if stats:
                    check_instance = check(**stats)
                else:
                    check_instance = check()
                for option_name, option_value in options.items():
                    setattr(check_instance, option_name, option_value)
                checks.append(check_instance)
            else:
                checks.append(check(stats))
        except TypeError:
            checks.append(check(stats))
    return checks if checks else None
