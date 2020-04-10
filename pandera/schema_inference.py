"""Module for inferring dataframe/series schema."""

import warnings
from typing import Any, Dict, Union

import pandas as pd

from .checks import Check
from .dtypes import PandasDtype
from .schemas import DataFrameSchema, SeriesSchema
from .schema_components import Column


NUMERIC_DTYPES = frozenset([
    PandasDtype.Float,
    PandasDtype.Float16,
    PandasDtype.Float32,
    PandasDtype.Float64,
    PandasDtype.Int,
    PandasDtype.Int8,
    PandasDtype.Int16,
    PandasDtype.Int32,
    PandasDtype.Int64,
    PandasDtype.UInt8,
    PandasDtype.UInt16,
    PandasDtype.UInt32,
    PandasDtype.UInt64,
])


def infer_schema(pandas_obj: Union[pd.DataFrame, pd.Series]):
    """Infer schema for pandas DataFrame or Series object.

    :param pandas_obj: DataFrame or Series object to infer.
    :returns: DataFrameSchema or SeriesSchema
    :raises: TypeError if pandas_obj is not expected type.
    """
    if isinstance(pandas_obj, pd.DataFrame):
        return infer_dataframe_schema(pandas_obj)
    elif isinstance(pandas_obj, pd.Series):
        return infer_series_schema(pandas_obj)
    else:
        raise TypeError(
            "pandas_obj type not recognized. Expected a pandas DataFrame or "
            "Series, found %s" % type(pandas_obj)
        )


def infer_dataframe_schema(df: pd.DataFrame) -> DataFrameSchema:
    """Infer a DataFrameSchema from a pandas DataFrame.

    :param df: DataFrame object to infer.
    :returns: DataFrameSchema
    """
    df_statistics = infer_dataframe_statistics(df)
    schema = DataFrameSchema(
        columns={
            colname: Column(
                properties["pandas_dtype"],
                checks=_parse_check_statistics(properties["checks"]),
                nullable=properties["nullable"],
            )
            for colname, properties in df_statistics["columns"].items()
        },
        coerce=True,
    )
    schema._is_inferred = True  # pylint: disable=protected-access
    return schema


def infer_series_schema(series) -> SeriesSchema:
    """Infer a SeriesSchema from a pandas DataFrame.

    :param series: Series object to infer.
    :returns: SeriesSchema
    """
    series_statistics = infer_series_statistics(series)
    schema = SeriesSchema(
        pandas_dtype=series_statistics["pandas_dtype"],
        checks=_parse_check_statistics(series_statistics["checks"]),
        nullable=series_statistics["nullable"],
        name=series_statistics["name"],
        coerce=True,
    )
    schema._is_inferred = True  # pylint: disable=protected-access
    return schema


def infer_dataframe_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Infer column and index statistics from a pandas DataFrame."""
    nullable_columns = df.isna().any()
    inferred_column_dtypes = pd.Series({
        col: _get_array_type(df[col]) for col in df
    })
    column_statistics = {
        col: {
            "pandas_dtype": dtype,
            "nullable": nullable_columns[col],
            "checks": _get_array_check_statistics(df[col], dtype),
        }
        for col, dtype in inferred_column_dtypes.iteritems()
    }
    return {
        "columns": column_statistics,
        "index": infer_index_statistics(df.index),
    }


def infer_series_statistics(series: pd.Series) -> Dict[str, Any]:
    """Infer column and index statistics from a pandas Series."""
    dtype = _get_array_type(series)
    return {
        "pandas_dtype": dtype,
        "nullable": series.isna().any(),
        "checks": _get_array_check_statistics(series, dtype),
        "name": series.name,
    }


def infer_index_statistics(index: Union[pd.Index, pd.MultiIndex]):
    """Infer index statistics given a pandas Index object."""

    def _index_stats(index_level):
        dtype = _get_array_type(index_level)
        return {
            "name": index_level.name,
            "pandas_dtype": dtype,
            "nullable": index_level.isna().any(),
            "checks": _get_array_check_statistics(index_level, dtype),
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
            "index type %s not recognized, skipping index inference" %
            type(index),
            UserWarning
        )
        index_statistics = []
    return index_statistics if index_statistics else None


def _get_array_type(x):
    # get most granular type possible
    dtype = PandasDtype.from_str_alias(str(x.dtype))
    # for object arrays, try to infer dtype
    if dtype is PandasDtype.Object:
        dtype = PandasDtype.from_pandas_api_type(
            pd.api.types.infer_dtype(x, skipna=True)
        )
    return dtype


def _get_array_check_statistics(
        x, dtype: PandasDtype) -> Union[Dict[str, Any], None]:
    """Get check statistics from an array-like object."""
    if dtype in NUMERIC_DTYPES or dtype is PandasDtype.DateTime:
        check_stats = {
            "min": x.min(),
            "max": x.max(),
        }
    elif dtype is PandasDtype.Category:
        try:
            categories = x.cat.categories
        except AttributeError:
            categories = x.categories
        check_stats = {
            "levels": categories.tolist(),
        }
    else:
        check_stats = {}
    return check_stats if check_stats else None


def _parse_check_statistics(check_stats: Union[Dict[str, Any], None]):
    """Convert check statistics to a list of Check objects."""
    if check_stats is None:
        return None
    checks = []
    if "min" in check_stats:
        checks.append(Check.greater_than_or_equal_to(check_stats["min"]))
    if "max" in check_stats:
        checks.append(Check.less_than_or_equal_to(check_stats["max"]))
    if "levels" in check_stats:
        checks.append(Check.isin(check_stats["levels"]))
    return checks if checks else None
