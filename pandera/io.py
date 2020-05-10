"""Module for reading and writing schema objects."""

import yaml
from pathlib import Path

import pandas as pd

from .dtypes import PandasDtype
from .schema_statistics import get_dataframe_schema_statistics


SCHEMA_TYPES = {"dataframe"}
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
NOT_JSON_SERIALIZABLE = {
    PandasDtype.DateTime, PandasDtype.Timedelta
}


def _serialize_check_stats(check_stats, pandas_dtype):
    """Serialize check statistics into json/yaml-compatible format."""

    def handle_stat_dtype(stat):
        if pandas_dtype == PandasDtype.DateTime:
            return stat.strftime(DATETIME_FORMAT)
        elif pandas_dtype == PandasDtype.Timedelta:
            # serialize to int in nanoseconds
            return stat.delta
        return stat

    # for unary checks, return a single value instead of a dictionary
    if len(check_stats) == 1:
        return handle_stat_dtype(list(check_stats.values())[0])

    # otherwise return a dictionary of keyword args needed to create the Check
    serialized_check_stats = {}
    for arg, stat in check_stats.items():
        serialized_check_stats[arg] = handle_stat_dtype(stat)
    return serialized_check_stats


def _serialize_component_stats(component_stats):
    """
    Serialize column or index statistics into json/yaml-compatible format.
    """
    serialized_checks = None
    if component_stats["checks"] is not None:
        serialized_checks = {
            check_name: _serialize_check_stats(
                check_stats, component_stats["pandas_dtype"]
            )
            for check_name, check_stats in component_stats["checks"].items()
        }
    return {
        "pandas_dtype": component_stats["pandas_dtype"].value,
        "nullable": component_stats["nullable"],
        "checks": serialized_checks,
        **(
            {} if "name" not in component_stats else
            {"name": component_stats["name"]}
        )
    }


def _serialize_schema(statistics):
    """Serialize dataframe schema into into json/yaml-compatible format."""
    from pandera import __version__  # pylint: disable-all

    columns, index = None, None
    if statistics["columns"] is not None:
        columns = {
            col_name: _serialize_component_stats(column_stats)
            for col_name, column_stats in statistics["columns"].items()
        }

    if statistics["index"] is not None:
        index = [
            _serialize_component_stats(index_stats)
            for index_stats in statistics["index"]
        ]

    return {
        "schema_type": "dataframe",
        "version": __version__,
        "columns": columns,
        "index": index,
    }


def _deserialize_check_stats(check, serialized_check_stats, pandas_dtype):

    def handle_stat_dtype(stat):
        if pandas_dtype == PandasDtype.DateTime:
            return pd.to_datetime(stat, format=DATETIME_FORMAT)
        elif pandas_dtype == PandasDtype.Timedelta:
            # serialize to int in nanoseconds
            return pd.to_timedelta(stat, unit="ns")
        return stat

    if isinstance(serialized_check_stats, dict):
        # handle case where serialized check stats are in the form of a
        # dictionary mapping Check arg names to values.
        check_stats = {}
        for arg, stat in serialized_check_stats.items():
            check_stats[arg] = handle_stat_dtype(stat)
        return check(**check_stats)
    # otherwise assume unary check function signature
    return check(handle_stat_dtype(serialized_check_stats))


def _deserialize_component_stats(serialized_component_stats):
    # pylint: disable-all
    from pandera import Check

    pandas_dtype = PandasDtype.from_str_alias(
        serialized_component_stats["pandas_dtype"]
    )
    checks = None
    if serialized_component_stats["checks"] is not None:
        checks = [
            _deserialize_check_stats(
                getattr(Check, check_name), check_stats, pandas_dtype
            )
            for check_name, check_stats
            in serialized_component_stats["checks"].items()
        ]
    return {
        "pandas_dtype": pandas_dtype,
        "nullable": serialized_component_stats["nullable"],
        "checks": checks,
        **(
            {} if "name" not in serialized_component_stats else
            {"name": serialized_component_stats["name"]}
        )
    }


def _deserialize_schema(serialized_schema):
    # pylint: disable-all
    from pandera import DataFrameSchema, Column, Index, MultiIndex

    columns, index = None, None
    if serialized_schema["columns"] is not None:
        columns = {
            col_name: Column(**_deserialize_component_stats(column_stats))
            for col_name, column_stats in serialized_schema["columns"].items()
        }

    if serialized_schema["index"] is not None:
        index = [
            _deserialize_component_stats(index_component)
            for index_component in serialized_schema["index"]
        ]

    if len(index) == 1:
        index = Index(**index[0])
    else:
        index = MultiIndex(indexes=[
            Index(**index_properties) for index_properties in index
        ])

    return DataFrameSchema(
        columns={
            col_name: column
            for col_name, column in columns.items()
        },
        index=index,
    )


def from_yaml(yaml_schema):
    """Create :py:class:`DataFrameSchema` from yaml file.

    :param yaml_schema: str or Path to yaml schema, or serialized yaml string.
    :returns: dataframe schema.
    """
    try:
        with Path(yaml_schema).open("r") as f:
            serialized_schema = yaml.safe_load(f)
    except (TypeError, OSError):
        serialized_schema = yaml.safe_load(yaml_schema)
    return _deserialize_schema(serialized_schema)


def to_yaml(dataframe_schema, stream=None):
    """Write :py:class:`DataFrameSchema` to yaml file.

    :param dataframe_schema: schema to write to file or dump to string.
    :param stream: file stream to write to. If None, dumps to string.
    :returns: yaml string if stream is None, otherwise returns None.
    """
    statistics = _serialize_schema(
        get_dataframe_schema_statistics(dataframe_schema))

    def _write_yaml(obj, stream):
        try:
            return yaml.safe_dump(obj, stream=stream, sort_keys=False)
        except TypeError:
            return yaml.safe_dump(obj, stream=stream)

    try:
        with Path(stream).open("w") as f:
            _write_yaml(statistics, f)
    except (TypeError, OSError):
        return _write_yaml(statistics, stream)
