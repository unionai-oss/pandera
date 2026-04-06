"""Serialize and deserialize polars :class:`~pandera.api.polars.container.DataFrameSchema`."""

from __future__ import annotations

import enum
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pandera import dtypes
from pandera.api.checks import Check
from pandera.api.polars.components import Column
from pandera.api.polars.container import DataFrameSchema
from pandera.engines import polars_engine
from pandera.errors import SchemaDefinitionError
from pandera.io._check_io import checks_dict_to_list
from pandera.io._constants import DATETIME_FORMAT, MISSING_PYYAML_MESSAGE
from pandera.io._minimal import apply_minimal_dataframe_container
from pandera.schema_statistics.polars import get_dataframe_schema_statistics


def _serialize_check_stats(check_stats, dtype=None):
    """Serialize check statistics into json/yaml-compatible format."""

    def handle_stat_dtype(stat):
        if isinstance(stat, type) and issubclass(stat, enum.Enum):
            return [e.value for e in stat]

        if dtype is not None:
            if polars_engine.Engine.dtype(dtypes.DateTime).check(
                dtype
            ) and hasattr(stat, "strftime"):
                return stat.strftime(DATETIME_FORMAT)
            if polars_engine.Engine.dtype(dtypes.Timedelta).check(dtype):
                return getattr(stat, "value", stat)

        return stat

    check_options = (
        check_stats.pop("options", {}) if isinstance(check_stats, dict) else {}
    )

    if isinstance(check_stats, dict) and len(check_stats) == 1:
        value = handle_stat_dtype(list(check_stats.values())[0])
        if check_options:
            return {"value": value, "options": check_options}
        return value

    if isinstance(check_stats, dict):
        serialized_check_stats = {}
        for arg, stat in check_stats.items():
            serialized_check_stats[arg] = handle_stat_dtype(stat)
        if check_options:
            serialized_check_stats["options"] = check_options
        return serialized_check_stats

    return handle_stat_dtype(check_stats)


def _serialize_dataframe_stats(dataframe_checks):
    serialized_checks = []
    for check_stats in dataframe_checks:
        serialized_check_stats = _serialize_check_stats(check_stats)
        serialized_checks.append(serialized_check_stats)
    return serialized_checks


def _serialize_component_stats(component_stats):
    """Serialize column statistics into json/yaml-compatible format."""
    serialized_checks = None
    if component_stats["checks"] is not None:
        serialized_checks = []
        for check_stats in component_stats["checks"]:
            serialized_check_stats = _serialize_check_stats(
                check_stats, component_stats["dtype"]
            )
            serialized_checks.append(serialized_check_stats)

    dtype = component_stats.get("dtype")
    if dtype:
        dtype = str(dtype)

    description = component_stats.get("description")
    title = component_stats.get("title")

    return {
        "title": title,
        "description": description,
        "dtype": dtype,
        "nullable": component_stats["nullable"],
        "checks": serialized_checks,
        **{
            key: component_stats.get(key)
            for key in [
                "name",
                "unique",
                "coerce",
                "required",
                "regex",
                "default",
                "report_duplicates",
                "drop_invalid_rows",
            ]
            if key in component_stats
        },
    }


def serialize_schema(
    dataframe_schema, *, minimal: bool = True
) -> dict[str, Any]:
    """Serialize a polars dataframe schema into a JSON/YAML-compatible dict.

    :param minimal: If True (default), omit keys equal to constructor defaults.
    """
    from pandera import __version__

    statistics = get_dataframe_schema_statistics(dataframe_schema)

    columns, checks = None, None
    if statistics["columns"] is not None:
        columns = {
            col_name: _serialize_component_stats(column_stats)
            for col_name, column_stats in statistics["columns"].items()
        }

    if statistics["checks"] is not None:
        checks = _serialize_dataframe_stats(statistics["checks"])

    out = {
        "schema_type": "polars_dataframe",
        "version": __version__,
        "columns": columns,
        "checks": checks,
        "index": None,
        "dtype": dataframe_schema.dtype,
        "coerce": dataframe_schema.coerce,
        "strict": dataframe_schema.strict,
        "name": dataframe_schema.name,
        "ordered": dataframe_schema.ordered,
        "unique": dataframe_schema.unique,
        "report_duplicates": dataframe_schema.report_duplicates,
        "unique_column_names": dataframe_schema.unique_column_names,
        "add_missing_columns": dataframe_schema.add_missing_columns,
        "title": dataframe_schema.title,
        "description": dataframe_schema.description,
    }
    if minimal:
        apply_minimal_dataframe_container(out, dataframe_schema)
    return out


def _deserialize_check_stats(check, serialized_check_stats, dtype=None):
    """Deserialize check statistics and reconstruct check with options."""

    def handle_stat_dtype(stat):
        if dtype is None:
            return stat
        try:
            import pandas as pd

            if polars_engine.Engine.dtype(dtypes.DateTime).check(dtype):
                return pd.to_datetime(stat, format=DATETIME_FORMAT)
            if polars_engine.Engine.dtype(dtypes.Timedelta).check(dtype):
                return pd.to_timedelta(stat, unit="ns")
        except (TypeError, ValueError):
            return stat
        return stat

    options = {}
    if isinstance(serialized_check_stats, dict):
        options = serialized_check_stats.pop("options", {})
        if (
            "value" in serialized_check_stats
            and len(serialized_check_stats) == 1
        ):
            serialized_check_stats = serialized_check_stats["value"]

    if isinstance(serialized_check_stats, dict):
        check_stats = {}
        for arg, stat in serialized_check_stats.items():
            check_stats[arg] = handle_stat_dtype(stat)
        check_instance = check(**check_stats)
    else:
        check_instance = check(handle_stat_dtype(serialized_check_stats))

    if options:
        for option_name, option_value in options.items():
            if option_name != "check_name":
                setattr(check_instance, option_name, option_value)

    return check_instance


def _deserialize_component_stats(serialized_component_stats):
    dtype = serialized_component_stats.get("dtype")
    if dtype:
        s = str(dtype)
        try:
            dtype = polars_engine.Engine.dtype(s)
        except TypeError:
            dtype = polars_engine.Engine.dtype(s.lower())

    description = serialized_component_stats.get("description")
    title = serialized_component_stats.get("title")

    checks = checks_dict_to_list(serialized_component_stats.get("checks"))
    if checks is not None:
        checks = [
            _deserialize_check_stats(
                getattr(Check, check["options"]["check_name"]), check, dtype
            )
            for check in checks
        ]
    return {
        "title": title,
        "description": description,
        "dtype": dtype,
        "checks": checks,
        **{
            key: serialized_component_stats.get(key)
            for key in [
                "name",
                "nullable",
                "unique",
                "coerce",
                "required",
                "regex",
                "default",
                "report_duplicates",
                "drop_invalid_rows",
            ]
            if key in serialized_component_stats
        },
    }


def deserialize_schema(serialized_schema):
    """Deserialize a mapping into :class:`~pandera.api.polars.container.DataFrameSchema`."""
    serialized_schema = serialized_schema if serialized_schema else {}

    if not isinstance(serialized_schema, Mapping):
        raise SchemaDefinitionError("Schema representation must be a mapping.")

    st = serialized_schema.get("schema_type")
    if st is not None and st != "polars_dataframe":
        raise SchemaDefinitionError(
            "Expected schema_type 'polars_dataframe' for polars IO, "
            f"got {st!r}. Use pandera.io.pandas_io for pandas schemas."
        )

    columns = serialized_schema.get("columns")
    checks = checks_dict_to_list(serialized_schema.get("checks"))

    if columns is not None:
        columns = {
            col_name: Column(**_deserialize_component_stats(column_stats))
            for col_name, column_stats in columns.items()
        }

    if checks is not None:
        checks = [
            _deserialize_check_stats(
                getattr(Check, check["options"]["check_name"]), check
            )
            for check in checks
        ]

    return DataFrameSchema(
        columns=columns,
        checks=checks,
        index=None,
        dtype=serialized_schema.get("dtype", None),
        coerce=serialized_schema.get("coerce", False),
        strict=serialized_schema.get("strict", False),
        name=serialized_schema.get("name", None),
        ordered=serialized_schema.get("ordered", False),
        unique=serialized_schema.get("unique", None),
        report_duplicates=serialized_schema.get("report_duplicates", "all"),
        unique_column_names=serialized_schema.get(
            "unique_column_names", False
        ),
        add_missing_columns=serialized_schema.get(
            "add_missing_columns", False
        ),
        title=serialized_schema.get("title", None),
        description=serialized_schema.get("description", None),
    )


def from_yaml(yaml_schema):
    """Load a polars :class:`DataFrameSchema` from YAML."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError(MISSING_PYYAML_MESSAGE) from exc

    try:
        with Path(yaml_schema).open("r", encoding="utf-8") as f:
            serialized_schema = yaml.safe_load(f)
    except (TypeError, OSError):
        serialized_schema = yaml.safe_load(yaml_schema)
    return deserialize_schema(serialized_schema)


def to_yaml(dataframe_schema, stream=None, *, minimal: bool = True):
    """Write a polars :class:`DataFrameSchema` to YAML."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError(MISSING_PYYAML_MESSAGE) from exc

    statistics = serialize_schema(dataframe_schema, minimal=minimal)

    def _write_yaml(obj, stream):
        return yaml.safe_dump(obj, stream=stream, sort_keys=False)

    try:
        with Path(stream).open("w", encoding="utf-8") as f:
            _write_yaml(statistics, f)
    except (TypeError, OSError):
        return _write_yaml(statistics, stream)


def from_json(source):
    """Load a polars :class:`DataFrameSchema` from JSON."""
    if isinstance(source, str):
        try:
            serialized_schema = json.loads(source)
        except json.decoder.JSONDecodeError:
            with Path(source).open(encoding="utf-8") as f:
                serialized_schema = json.load(fp=f)
    elif isinstance(source, Path):
        with source.open(encoding="utf-8") as f:
            serialized_schema = json.load(fp=f)
    else:
        serialized_schema = json.load(fp=source)

    return deserialize_schema(serialized_schema)


def to_json(dataframe_schema, target=None, *, minimal: bool = True, **kwargs):
    """Write a polars :class:`DataFrameSchema` to JSON."""
    serialized_schema = serialize_schema(dataframe_schema, minimal=minimal)

    if target is None:
        return json.dumps(serialized_schema, sort_keys=False, **kwargs)

    if isinstance(target, (str, Path)):
        with Path(target).open("w", encoding="utf-8") as f:
            json.dump(serialized_schema, fp=f, sort_keys=False, **kwargs)
    else:
        json.dump(serialized_schema, fp=target, sort_keys=False, **kwargs)
