"""Module for inferring the statistics of pandas objects."""

import warnings
from typing import Any, Dict, Union

import pandas as pd

from pandera import dtypes
from pandera.api.checks import Check
from pandera.engines import pandas_engine


def infer_dataframe_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Infer column and index statistics from a pandas DataFrame."""
    nullable_columns = df.isna().any()
    inferred_column_dtypes = {col: _get_array_type(df[col]) for col in df}
    column_statistics = {
        col: {
            "dtype": dtype,
            "nullable": bool(nullable_columns[col]),  # type: ignore
            "checks": _get_array_check_statistics(df[col], dtype),
        }
        for col, dtype in inferred_column_dtypes.items()
    }
    return {
        "columns": column_statistics if column_statistics else None,
        "index": infer_index_statistics(df.index),
    }


def infer_series_statistics(series: pd.Series) -> Dict[str, Any]:
    """Infer column and index statistics from a pandas Series."""
    dtype = _get_array_type(series)
    return {
        "dtype": dtype,
        "nullable": bool(series.isna().any()),
        "checks": _get_array_check_statistics(series, dtype),
        "name": series.name,
    }


def infer_index_statistics(index: Union[pd.Index, pd.MultiIndex]):
    """Infer index statistics given a pandas Index object."""

    def _index_stats(index_level):
        dtype = _get_array_type(index_level)
        return {
            "dtype": dtype,
            "nullable": bool(index_level.isna().any()),
            "checks": _get_array_check_statistics(index_level, dtype),
            "name": index_level.name,
        }

    if isinstance(index, pd.MultiIndex):
        index_statistics = [
            _index_stats(index.get_level_values(i))
            for i in range(index.nlevels)
        ]
    elif isinstance(index, pd.Index):
        index_statistics = [_index_stats(index)]
    else:
        warnings.warn(
            f"index type {type(index)} not recognized, skipping index inference",
            UserWarning,
        )
        index_statistics = []
    return index_statistics if index_statistics else None


def parse_check_statistics(check_stats: Union[Dict[str, Any], None]):
    """Convert check statistics to a list of Check objects."""
    if check_stats is None:
        return None
    checks = []
    for check_name, stats in check_stats.items():
        check = getattr(Check, check_name)
        try:
            checks.append(check(**stats))
        except TypeError:
            # if stats cannot be unpacked as key-word args, assume unary check.
            checks.append(check(stats))
    return checks if checks else None


def get_dataframe_schema_statistics(dataframe_schema):
    """Get statistical properties from dataframe schema."""
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
            }
            for col_name, column in dataframe_schema.columns.items()
        },
        "checks": parse_checks(dataframe_schema.checks),
        "index": (
            None
            if dataframe_schema.index is None
            else get_index_schema_statistics(dataframe_schema.index)
        ),
        "coerce": dataframe_schema.coerce,
    }
    return statistics


def _get_series_base_schema_statistics(series_schema_base):
    return {
        "dtype": series_schema_base.dtype,
        "nullable": series_schema_base.nullable,
        "checks": parse_checks(series_schema_base.checks),
        "coerce": series_schema_base.coerce,
        "name": series_schema_base.name,
        "unique": series_schema_base.unique,
        "title": series_schema_base.title,
        "description": series_schema_base.description,
    }


def get_index_schema_statistics(index_schema_component):
    """Get statistical properties of index schema component."""
    try:
        # get index components from MultiIndex
        index_components = index_schema_component.indexes
    except AttributeError:
        index_components = [index_schema_component]
    return [
        _get_series_base_schema_statistics(index_component)
        for index_component in index_components
    ]


def get_series_schema_statistics(series_schema):
    """Get statistical properties from series schema."""
    return _get_series_base_schema_statistics(series_schema)


def parse_checks(checks) -> Union[Dict[str, Any], None]:
    """Convert Check object to check statistics."""
    check_statistics = {}
    _check_memo = {}
    for check in checks:
        if check not in Check:
            warnings.warn(
                "Only registered checks may be serialized to statistics. "
                "Did you forget to register it with the extension API? "
                f"Check `{check.name}` will be skipped."
            )
            continue

        check_statistics[check.name] = (
            {} if check.statistics is None else check.statistics
        )
        _check_memo[check.name] = check

    # raise ValueError on incompatible checks
    if (
        "greater_than_or_equal_to" in check_statistics
        and "less_than_or_equal_to" in check_statistics
    ):
        min_value = check_statistics.get(
            "greater_than_or_equal_to", float("-inf")
        )["min_value"]
        max_value = check_statistics.get(
            "less_than_or_equal_to", float("inf")
        )["max_value"]
        if min_value > max_value:
            raise ValueError(
                f"checks {_check_memo['greater_than_or_equal_to']} "
                f"and {_check_memo['less_than_or_equal_to']} are incompatible, reason: "
                f"min value {min_value} > max value {max_value}"
            )
    return check_statistics if check_statistics else None


def _get_array_type(x):
    # get most granular type possible

    data_type = pandas_engine.Engine.dtype(x.dtype)
    # for object arrays, try to infer dtype
    if data_type is pandas_engine.Engine.dtype("object"):
        inferred_alias = pd.api.types.infer_dtype(x, skipna=False)
        if inferred_alias != "string":
            data_type = pandas_engine.Engine.dtype(inferred_alias)
    return data_type


def _get_array_check_statistics(
    x, data_type: dtypes.DataType
) -> Union[Dict[str, Any], None]:
    """Get check statistics from an array-like object."""
    if x.isna().all():
        return None
    if dtypes.is_datetime(data_type):
        check_stats = {
            "greater_than_or_equal_to": x.min(),
            "less_than_or_equal_to": x.max(),
        }
    elif dtypes.is_numeric(data_type) and not dtypes.is_bool(data_type):
        check_stats = {
            "greater_than_or_equal_to": float(x.min()),
            "less_than_or_equal_to": float(x.max()),
        }
    elif dtypes.is_category(data_type):
        try:
            categories = x.cat.categories
        except AttributeError:
            categories = x.categories
        check_stats = {
            "isin": categories.tolist(),
        }
    else:
        check_stats = {}
    return check_stats if check_stats else None
