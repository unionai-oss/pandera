"""Module for reading and writing schema objects."""

import warnings
from functools import partial
from pathlib import Path

import pandas as pd

from .dtypes import PandasDtype
from .schema_statistics import get_dataframe_schema_statistics

try:
    import black
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        'IO and formatting requires "pyyaml" and "black" to be installed. \n'
        "You can install pandera together with the IO dependencies with: \n"
        "pip install pandera[io]\n"
    ) from exc


SCHEMA_TYPES = {"dataframe"}
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
NOT_JSON_SERIALIZABLE = {PandasDtype.DateTime, PandasDtype.Timedelta}


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
        serialized_checks = {}
        for check_name, check_stats in component_stats["checks"].items():
            if check_stats is None:
                warnings.warn(
                    f"Check {check_name} cannot be serialized. This check will be "
                    f"ignored"
                )
            else:
                serialized_checks[check_name] = _serialize_check_stats(
                    check_stats, component_stats["pandas_dtype"]
                )
    return {
        "pandas_dtype": component_stats["pandas_dtype"].value,
        "nullable": component_stats["nullable"],
        "checks": serialized_checks,
        **{
            key: component_stats.get(key)
            for key in [
                "name",
                "allow_duplicates",
                "coerce",
                "required",
                "regex",
            ]
            if key in component_stats
        },
    }


def _serialize_schema(dataframe_schema):
    """Serialize dataframe schema into into json/yaml-compatible format."""
    from pandera import __version__  # pylint: disable=import-outside-toplevel

    statistics = get_dataframe_schema_statistics(dataframe_schema)

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
        "coerce": dataframe_schema.coerce,
        "strict": dataframe_schema.strict,
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
    from pandera import Check  # pylint: disable=import-outside-toplevel

    pandas_dtype = PandasDtype.from_str_alias(
        serialized_component_stats["pandas_dtype"]
    )
    checks = None
    if serialized_component_stats["checks"] is not None:
        checks = [
            _deserialize_check_stats(
                getattr(Check, check_name), check_stats, pandas_dtype
            )
            for check_name, check_stats in serialized_component_stats[
                "checks"
            ].items()
        ]
    return {
        "pandas_dtype": pandas_dtype,
        "nullable": serialized_component_stats["nullable"],
        "checks": checks,
        **{
            key: serialized_component_stats.get(key)
            for key in [
                "name",
                "allow_duplicates",
                "coerce",
                "required",
                "regex",
            ]
            if key in serialized_component_stats
        },
    }


def _deserialize_schema(serialized_schema):
    # pylint: disable=import-outside-toplevel
    from pandera import Column, DataFrameSchema, Index, MultiIndex

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

    if index is None:
        pass
    elif len(index) == 1:
        index = Index(**index[0])
    else:
        index = MultiIndex(
            indexes=[Index(**index_properties) for index_properties in index]
        )

    return DataFrameSchema(
        columns=columns,
        index=index,
        coerce=serialized_schema["coerce"],
        strict=serialized_schema["strict"],
    )


def from_yaml(yaml_schema):
    """Create :class:`~pandera.schemas.DataFrameSchema` from yaml file.

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
    """Write :class:`~pandera.schemas.DataFrameSchema` to yaml file.

    :param dataframe_schema: schema to write to file or dump to string.
    :param stream: file stream to write to. If None, dumps to string.
    :returns: yaml string if stream is None, otherwise returns None.
    """
    statistics = _serialize_schema(dataframe_schema)

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


SCRIPT_TEMPLATE = """
from pandera import (
    DataFrameSchema, Column, Check, Index, MultiIndex, PandasDtype
)

schema = DataFrameSchema(
    columns={{{columns}}},
    index={index},
    coerce={coerce},
    strict={strict},
    name={name},
)
"""

COLUMN_TEMPLATE = """
Column(
    pandas_dtype={pandas_dtype},
    checks={checks},
    nullable={nullable},
    allow_duplicates={allow_duplicates},
    coerce={coerce},
    required={required},
    regex={regex},
)
"""

INDEX_TEMPLATE = (
    "Index(pandas_dtype={pandas_dtype},checks={checks},"
    "nullable={nullable},coerce={coerce},name={name})"
)

MULTIINDEX_TEMPLATE = """
MultiIndex(indexes=[{indexes}])
"""


def _format_checks(checks_dict):
    if checks_dict is None:
        return "None"

    checks = []
    for check_name, check_kwargs in checks_dict.items():
        if check_kwargs is None:
            warnings.warn(
                f"Check {check_name} cannot be serialized. This check will be ignored"
            )
        else:
            args = ", ".join(
                "{}={}".format(k, v.__repr__())
                for k, v in check_kwargs.items()
            )
            checks.append(f"Check.{check_name}({args})")
    return f"[{', '.join(checks)}]"


def _format_index(index_statistics):
    index = []
    for properties in index_statistics:
        index_code = INDEX_TEMPLATE.format(
            pandas_dtype=f"PandasDtype.{properties['pandas_dtype'].name}",
            checks=(
                "None"
                if properties["checks"] is None
                else _format_checks(properties["checks"])
            ),
            nullable=properties["nullable"],
            coerce=properties["coerce"],
            name=(
                "None"
                if properties["name"] is None
                else f"\"{properties['name']}\""
            ),
        )
        index.append(index_code.strip())

    if len(index) == 1:
        return index[0]

    return MULTIINDEX_TEMPLATE.format(indexes=",".join(index)).strip()


def _format_script(script):
    formatter = partial(black.format_str, mode=black.FileMode(line_length=80))
    return formatter(script)


def to_script(dataframe_schema, path_or_buf=None):
    """Write :class:`~pandera.schemas.DataFrameSchema` to a python script.

    :param dataframe_schema: schema to write to file or dump to string.
    :param path_or_buf: filepath or buf stream to write to. If None, outputs
        string representation of the script.
    :returns: yaml string if stream is None, otherwise returns None.
    """
    statistics = get_dataframe_schema_statistics(dataframe_schema)

    columns = {}
    for colname, properties in statistics["columns"].items():
        column_code = COLUMN_TEMPLATE.format(
            pandas_dtype=f"PandasDtype.{properties['pandas_dtype'].name}",
            checks=_format_checks(properties["checks"]),
            nullable=properties["nullable"],
            allow_duplicates=properties["allow_duplicates"],
            coerce=properties["coerce"],
            required=properties["required"],
            regex=properties["regex"],
        )
        columns[colname] = column_code.strip()

    index = (
        None
        if statistics["index"] is None
        else _format_index(statistics["index"])
    )

    column_str = ", ".join("'{}': {}".format(k, v) for k, v in columns.items())

    script = SCRIPT_TEMPLATE.format(
        columns=column_str,
        index=index,
        coerce=dataframe_schema.coerce,
        strict=dataframe_schema.strict,
        name=dataframe_schema.name.__repr__(),
    ).strip()

    # add pandas imports to handle datetime and timedelta.
    if "Timedelta" in script:
        script = "from pandas import Timedelta\n" + script
    if "Timestamp" in script:
        script = "from pandas import Timestamp\n" + script

    formatted_script = _format_script(script)

    if path_or_buf is None:
        return formatted_script

    with Path(path_or_buf).open("w") as f:
        f.write(formatted_script)
