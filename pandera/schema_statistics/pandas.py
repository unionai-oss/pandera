"""Module for inferring the statistics of pandas objects."""

import warnings
from typing import Any, Union

import pandas as pd

from pandera import dtypes
from pandera.engines import pandas_engine
from pandera.schema_statistics.common import (
    parse_check_statistics,  # noqa: F401  (re-exported for compatibility)
    parse_checks,
    string_length_check_statistics,  # noqa: F401  (re-exported)
)


def infer_dataframe_statistics(
    df: pd.DataFrame, *, infer_str_length: bool = False
) -> dict[str, Any]:
    """Infer column and index statistics from a pandas DataFrame.

    :param infer_str_length: also infer ``str_length`` checks for
        string-like columns. Off by default so that inferred schemas keep
        validating after a column's dtype is updated (e.g. together with a
        parser).
    """
    nullable_columns = df.isna().any()
    inferred_column_dtypes = {col: _get_array_type(df[col]) for col in df}
    column_statistics = {
        col: {
            "dtype": dtype,
            "nullable": bool(nullable_columns[col]),  # type: ignore
            "checks": _get_array_check_statistics(
                df[col], dtype, infer_str_length=infer_str_length
            ),
        }
        for col, dtype in inferred_column_dtypes.items()
    }
    return {
        "columns": column_statistics if column_statistics else None,
        "index": infer_index_statistics(df.index),
    }


def infer_series_statistics(
    series: pd.Series, *, infer_str_length: bool = False
) -> dict[str, Any]:
    """Infer column and index statistics from a pandas Series.

    :param infer_str_length: also infer ``str_length`` checks for
        string-like values (see :func:`infer_dataframe_statistics`).
    """
    dtype = _get_array_type(series)
    return {
        "dtype": dtype,
        "nullable": bool(series.isna().any()),
        "checks": _get_array_check_statistics(
            series, dtype, infer_str_length=infer_str_length
        ),
        "name": series.name,
    }


def infer_index_statistics(index: Union[pd.Index, pd.MultiIndex]):
    """Infer index statistics given a pandas Index object.

    Only dtype, nullability, and level names are inferred; index levels do not
    receive inferred checks (unlike dataframe columns).
    """

    def _index_stats(index_level):
        dtype = _get_array_type(index_level)
        return {
            "dtype": dtype,
            "nullable": bool(index_level.isna().any()),
            # Index dtypes are inferred only; do not attach min/max, str_length,
            # isin, etc. (unlike columns).
            "checks": None,
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
                "drop_invalid_rows": column.drop_invalid_rows,
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
        "drop_invalid_rows": series_schema_base.drop_invalid_rows,
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


def _get_array_type(x):
    # get most granular type possible

    data_type = pandas_engine.Engine.dtype(x.dtype)
    # for object arrays, try to infer dtype
    if data_type is pandas_engine.Engine.dtype("object"):
        inferred_alias = pd.api.types.infer_dtype(x, skipna=False)
        if inferred_alias != "string":
            data_type = pandas_engine.Engine.dtype(inferred_alias)
    return data_type


def _should_infer_str_length(x: pd.Series, data_type: dtypes.DataType) -> bool:
    """True if the series is string-like (not categorical or numeric, etc.)."""
    if dtypes.is_category(data_type):
        return False
    if dtypes.is_numeric(data_type) or dtypes.is_bool(data_type):
        return False
    if dtypes.is_datetime(data_type) or dtypes.is_timedelta(data_type):
        return False
    if dtypes.is_binary(data_type):
        return False
    if dtypes.is_string(data_type):
        return True
    inferred = pd.api.types.infer_dtype(x, skipna=True)
    return inferred == "string"


def _string_length_bounds(x: pd.Series) -> tuple[int, int] | None:
    """Min and max string length over non-null values."""
    vals = x.dropna()
    if vals.empty:
        return None
    try:
        lens = vals.str.len()
    except (AttributeError, TypeError):
        return None
    if lens.isna().any():
        return None
    return int(lens.min()), int(lens.max())


def _get_array_check_statistics(
    x, data_type: dtypes.DataType, *, infer_str_length: bool = False
) -> Union[dict[str, Any], None]:
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
    elif infer_str_length and _should_infer_str_length(x, data_type):
        bounds = _string_length_bounds(x)
        if bounds is None:
            check_stats = {}
        else:
            check_stats = string_length_check_statistics(*bounds)
    else:
        check_stats = {}
    return check_stats if check_stats else None
