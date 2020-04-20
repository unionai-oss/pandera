"""Module for reading and writing schema objects."""

from pathlib import Path
import yaml

from .dtypes import PandasDtype
from .schema_statistics import get_dataframe_schema_statistics


SCHEMA_TYPES = {"dataframe"}


def _serialize_schema(statistics):
    from pandera import __version__  # pylint: disable-all

    columns, index = None, None
    if statistics["columns"] is not None:
        columns = {
            col_name: {
                k: v if k != "pandas_dtype" else v.value
                for k, v in column_stats.items()
            }
            for col_name, column_stats in statistics["columns"].items()
        }

    if statistics["index"] is not None:
        index = [
            {k: v if k != "pandas_dtype" else v.value
             for k, v in index_component.items()}
            for index_component in statistics["index"]
        ]

    return {
        "schema_type": "dataframe",
        "version": __version__,
        "columns": columns,
        "index": index,
    }


def _deserialize_schema(serialized_schema):
    # pylint: disable-all
    from pandera import DataFrameSchema, Column, Index, MultiIndex

    columns, index = None, None
    if serialized_schema["columns"] is not None:
        columns = {
            col_name: {
                k: v if k != "pandas_dtype" else PandasDtype.from_str_alias(v)
                for k, v in column_stats.items()
            }
            for col_name, column_stats in serialized_schema["columns"].items()
        }

    if serialized_schema["index"] is not None:
        index = [
            {k: v if k != "pandas_dtype" else PandasDtype.from_str_alias(v)
             for k, v in index_component.items()}
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
            col_name: Column(**col_properties)
            for col_name, col_properties in columns.items()
        },
        index=index,
    )


def from_yaml(yaml_schema):
    """Create :py:class:`DataFrameSchema` from yaml file.

    :param yaml_schema: str or Path to yaml schema, or serialized yaml string.
    :returns: dataframe schema.
    """
    try:
        with open(yaml_schema, "r") as f:
            serialized_schema = yaml.safe_load(f)
    except OSError:
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
