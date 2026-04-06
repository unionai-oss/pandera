"""Module for reading and writing xarray schema objects."""

from __future__ import annotations

import datetime
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from pandera.api.checks import Check
from pandera.io._constants import DATETIME_FORMAT, MISSING_PYYAML_MESSAGE
from pandera.io._minimal import apply_minimal_data_array, apply_minimal_dataset

# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _handle_stat_dtype_serialize(stat, dtype=None):
    """Make a check-statistic value JSON/YAML-safe.

    Mirrors ``handle_stat_dtype`` inside
    :func:`pandera.io.pandas_io._serialize_check_stats`:

    * ``datetime`` / ``datetime64`` → ISO format string
    * ``timedelta`` / ``timedelta64`` → integer nanoseconds
    * ``numpy`` scalars → native Python scalars
    """
    if isinstance(stat, (np.datetime64, datetime.datetime)):
        try:
            if isinstance(stat, np.datetime64):
                stat = stat.astype("datetime64[ns]").item()
            return stat.strftime(DATETIME_FORMAT)
        except (AttributeError, ValueError, OSError):
            return str(stat)

    if isinstance(stat, (np.timedelta64, datetime.timedelta)):
        try:
            if isinstance(stat, np.timedelta64):
                return int(stat / np.timedelta64(1, "ns"))
            return int(stat.total_seconds() * 1e9)
        except (TypeError, ValueError):
            return str(stat)

    if isinstance(stat, np.generic):
        return stat.item()

    return stat


def _handle_stat_dtype_deserialize(stat, dtype=None):
    """Reverse :func:`_handle_stat_dtype_serialize`.

    Mirrors ``handle_stat_dtype`` inside
    :func:`pandera.io.pandas_io._deserialize_check_stats`.
    """
    if dtype is None:
        return stat

    try:
        np_dtype = np.dtype(dtype)
    except TypeError:
        return stat

    if np.issubdtype(np_dtype, np.datetime64):
        try:
            dt = datetime.datetime.strptime(stat, DATETIME_FORMAT)
            return np.datetime64(dt, "ns")
        except (TypeError, ValueError):
            return stat

    if np.issubdtype(np_dtype, np.timedelta64):
        try:
            return np.timedelta64(int(stat), "ns")
        except (TypeError, ValueError):
            return stat

    return stat


def _serialize_check_stats(check_stats, dtype=None):
    """Serialize check statistics into json/yaml-compatible format.

    Matches the structure produced by
    :func:`pandera.io.pandas_io._serialize_check_stats`.
    """
    check_options = (
        check_stats.pop("options", {}) if isinstance(check_stats, dict) else {}
    )

    if isinstance(check_stats, dict) and len(check_stats) == 1:
        value = _handle_stat_dtype_serialize(
            list(check_stats.values())[0], dtype
        )
        if check_options:
            return {"value": value, "options": check_options}
        return value

    if isinstance(check_stats, dict):
        serialized_check_stats: dict[str, Any] = {}
        for arg, stat in check_stats.items():
            serialized_check_stats[arg] = _handle_stat_dtype_serialize(
                stat, dtype
            )
        if check_options:
            serialized_check_stats["options"] = check_options
        return serialized_check_stats

    return _handle_stat_dtype_serialize(check_stats, dtype)


def _serialize_checks(checks, dtype=None):
    """Serialize a list of check statistics.

    Parallels :func:`pandera.io.pandas_io._serialize_dataframe_stats`.
    """
    if checks is None:
        return None
    serialized = []
    for check_stats in checks:
        serialized.append(_serialize_check_stats(check_stats, dtype))
    return serialized if serialized else None


def _serialize_component_stats(component_stats):
    """Serialize a single component (DataVar / Coordinate) statistics dict.

    Parallels :func:`pandera.io.pandas_io._serialize_component_stats`.
    """
    serialized_checks = None
    if component_stats.get("checks") is not None:
        serialized_checks = []
        for check_stats in component_stats["checks"]:
            serialized_checks.append(
                _serialize_check_stats(
                    check_stats, component_stats.get("dtype")
                )
            )

    dtype = component_stats.get("dtype")
    if dtype is not None:
        dtype = str(dtype)

    result: dict[str, Any] = {
        "title": component_stats.get("title"),
        "description": component_stats.get("description"),
        "dtype": dtype,
        "nullable": component_stats.get("nullable"),
        "checks": serialized_checks,
    }

    for key in [
        "dims",
        "coerce",
        "required",
        "regex",
        "alias",
        "name",
    ]:
        if key in component_stats:
            val = component_stats[key]
            if key == "dims" and val is not None:
                val = list(val)
            result[key] = val

    return result


def _serialize_coord_stats(coord_stats):
    """Serialize coordinate statistics."""
    if coord_stats is None:
        return None
    serialized = {}
    for name, stats in coord_stats.items():
        if not stats:
            serialized[name] = {}
            continue
        serialized[name] = _serialize_component_stats(stats)
    return serialized if serialized else None


# ---------------------------------------------------------------------------
# Schema serialization
# ---------------------------------------------------------------------------


def serialize_data_array_schema(
    data_array_schema, *, minimal: bool = True
) -> dict[str, Any]:
    """Serialize a DataArraySchema into a json/yaml-compatible dict.

    :param data_array_schema: the schema to serialize.
    :param minimal: If True (default), omit keys equal to constructor defaults.
    :returns: dict representation of the schema.
    """
    from pandera import __version__
    from pandera.schema_statistics.xarray import (
        get_data_array_schema_statistics,
    )

    stats = get_data_array_schema_statistics(data_array_schema)

    out = {
        "schema_type": "data_array",
        "version": __version__,
        "dtype": stats["dtype"],
        "dims": list(stats["dims"]) if stats["dims"] else None,
        "ordered_dims": stats["ordered_dims"],
        "sizes": stats["sizes"],
        "shape": list(stats["shape"]) if stats["shape"] else None,
        "name": stats["name"],
        "nullable": stats["nullable"],
        "coerce": stats["coerce"],
        "coords": _serialize_coord_stats(stats["coords"]),
        "checks": _serialize_checks(stats["checks"]),
        "title": stats.get("title"),
        "description": stats.get("description"),
    }
    if minimal:
        apply_minimal_data_array(out, data_array_schema)
    return out


def serialize_dataset_schema(
    dataset_schema, *, minimal: bool = True
) -> dict[str, Any]:
    """Serialize a DatasetSchema into a json/yaml-compatible dict.

    :param dataset_schema: the schema to serialize.
    :param minimal: If True (default), omit keys equal to constructor defaults.
    :returns: dict representation of the schema.
    """
    from pandera import __version__
    from pandera.schema_statistics.xarray import (
        get_dataset_schema_statistics,
    )

    stats = get_dataset_schema_statistics(dataset_schema)

    data_vars = None
    if stats["data_vars"]:
        data_vars = {}
        for key, var_stats in stats["data_vars"].items():
            data_vars[key] = _serialize_component_stats(var_stats)

    out = {
        "schema_type": "dataset",
        "version": __version__,
        "data_vars": data_vars,
        "coords": _serialize_coord_stats(stats.get("coords")),
        "dims": list(stats["dims"]) if stats["dims"] else None,
        "ordered_dims": stats["ordered_dims"],
        "sizes": stats["sizes"],
        "strict": stats["strict"],
        "strict_coords": stats["strict_coords"],
        "strict_attrs": stats["strict_attrs"],
        "checks": _serialize_checks(stats["checks"]),
        "title": stats.get("title"),
        "description": stats.get("description"),
    }
    if minimal:
        apply_minimal_dataset(out, dataset_schema)
    return out


def serialize_schema(schema, *, minimal: bool = True) -> dict[str, Any]:
    """Serialize a DataArraySchema or DatasetSchema.

    :param schema: the schema to serialize.
    :param minimal: passed to ``serialize_*_schema`` functions.
    :returns: dict representation of the schema.
    """
    from pandera.api.xarray.container import DataArraySchema, DatasetSchema

    if isinstance(schema, DataArraySchema):
        return serialize_data_array_schema(schema, minimal=minimal)
    elif isinstance(schema, DatasetSchema):
        return serialize_dataset_schema(schema, minimal=minimal)
    else:
        raise TypeError(
            f"Expected DataArraySchema or DatasetSchema, got {type(schema)}"
        )


# ---------------------------------------------------------------------------
# Deserialization helpers
# ---------------------------------------------------------------------------


def _deserialize_check_stats(check, serialized_check_stats, dtype=None):
    """Deserialize check statistics and reconstruct the check.

    Mirrors :func:`pandera.io.pandas_io._deserialize_check_stats`.
    """
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
            check_stats[arg] = _handle_stat_dtype_deserialize(stat, dtype)
        check_instance = check(**check_stats)
    else:
        check_instance = check(
            _handle_stat_dtype_deserialize(serialized_check_stats, dtype)
        )

    if options:
        for option_name, option_value in options.items():
            if option_name != "check_name":
                setattr(check_instance, option_name, option_value)

    return check_instance


def _deserialize_checks(serialized_checks, dtype=None):
    """Deserialize a list of check statistics into Check objects.

    Parallels the check-deserialization loop in
    :func:`pandera.io.pandas_io._deserialize_component_stats`.
    """
    if serialized_checks is None:
        return None

    if isinstance(serialized_checks, dict):
        checks_list: list[Any] = []
        for check_name, check in serialized_checks.items():
            if not isinstance(check, dict):
                check = {"value": check}
            if "options" not in check:
                check["options"] = {}
            check["options"]["check_name"] = check_name
            checks_list.append(check)
        serialized_checks = checks_list

    checks: list[Check] = []
    for check_entry in serialized_checks:
        if not isinstance(check_entry, dict):
            continue
        options = check_entry.get("options", {})
        check_name = options.get("check_name")
        if check_name is None:
            warnings.warn(
                "Check entry missing 'check_name' in options, skipping."
            )
            continue
        check_fn = getattr(Check, check_name, None)
        if check_fn is None:
            warnings.warn(f"Check `{check_name}` not found, skipping.")
            continue
        checks.append(_deserialize_check_stats(check_fn, check_entry, dtype))

    return checks if checks else None


def _deserialize_component_stats(serialized_component_stats):
    """Deserialize a component statistics dict.

    Parallels :func:`pandera.io.pandas_io._deserialize_component_stats`.
    """
    dtype = serialized_component_stats.get("dtype")
    checks = _deserialize_checks(
        serialized_component_stats.get("checks"), dtype
    )
    result = {
        "title": serialized_component_stats.get("title"),
        "description": serialized_component_stats.get("description"),
        "dtype": dtype,
        "checks": checks,
    }
    for key in ["name", "nullable", "coerce", "required", "regex", "alias"]:
        if key in serialized_component_stats:
            result[key] = serialized_component_stats[key]
    if "dims" in serialized_component_stats:
        dims = serialized_component_stats["dims"]
        result["dims"] = tuple(dims) if dims is not None else None
    return result


def _deserialize_coord_stats(serialized_coords):
    """Deserialize coordinate statistics into Coordinate objects."""
    from pandera.api.xarray.components import Coordinate

    if serialized_coords is None:
        return None

    coords = {}
    for name, stats in serialized_coords.items():
        if not stats:
            coords[name] = Coordinate()
            continue
        deserialized = _deserialize_component_stats(stats)
        coords[name] = Coordinate(
            dtype=deserialized.get("dtype"),
            nullable=deserialized.get("nullable", False),
            checks=deserialized.get("checks"),
            title=deserialized.get("title"),
            description=deserialized.get("description"),
        )
    return coords if coords else None


# ---------------------------------------------------------------------------
# Schema deserialization
# ---------------------------------------------------------------------------


def deserialize_data_array_schema(serialized_schema):
    """Deserialize a dict into a DataArraySchema.

    :param serialized_schema: dict representation of the schema.
    :returns: DataArraySchema
    """
    from pandera.api.xarray.container import DataArraySchema

    dims = serialized_schema.get("dims")
    if dims is not None:
        dims = tuple(dims)

    shape = serialized_schema.get("shape")
    if shape is not None:
        shape = tuple(shape)

    return DataArraySchema(
        dtype=serialized_schema.get("dtype"),
        dims=dims,
        ordered_dims=serialized_schema.get("ordered_dims", True),
        sizes=serialized_schema.get("sizes"),
        shape=shape,
        coords=_deserialize_coord_stats(serialized_schema.get("coords")),
        name=serialized_schema.get("name"),
        nullable=serialized_schema.get("nullable", False),
        coerce=serialized_schema.get("coerce", False),
        checks=_deserialize_checks(serialized_schema.get("checks")),
        title=serialized_schema.get("title"),
        description=serialized_schema.get("description"),
    )


def deserialize_dataset_schema(serialized_schema):
    """Deserialize a dict into a DatasetSchema.

    :param serialized_schema: dict representation of the schema.
    :returns: DatasetSchema
    """
    from pandera.api.xarray.components import DataVar
    from pandera.api.xarray.container import DatasetSchema

    data_vars = None
    serialized_data_vars = serialized_schema.get("data_vars")
    if serialized_data_vars is not None:
        data_vars = {}
        for key, var_stats in serialized_data_vars.items():
            deserialized = _deserialize_component_stats(var_stats)
            dims = deserialized.get("dims")
            data_vars[key] = DataVar(
                dtype=deserialized.get("dtype"),
                dims=dims,
                nullable=deserialized.get("nullable", False),
                coerce=deserialized.get("coerce", False),
                required=deserialized.get("required", True),
                regex=deserialized.get("regex", False),
                alias=deserialized.get("alias"),
                checks=deserialized.get("checks"),
                title=deserialized.get("title"),
                description=deserialized.get("description"),
            )

    dims = serialized_schema.get("dims")
    if dims is not None:
        dims = tuple(dims)

    return DatasetSchema(
        data_vars=data_vars,
        coords=_deserialize_coord_stats(serialized_schema.get("coords")),
        dims=dims,
        ordered_dims=serialized_schema.get("ordered_dims", True),
        sizes=serialized_schema.get("sizes"),
        strict=serialized_schema.get("strict", False),
        strict_coords=serialized_schema.get("strict_coords", False),
        strict_attrs=serialized_schema.get("strict_attrs", False),
        checks=_deserialize_checks(serialized_schema.get("checks")),
        name=serialized_schema.get("name"),
        title=serialized_schema.get("title"),
        description=serialized_schema.get("description"),
    )


def deserialize_schema(serialized_schema):
    """Deserialize a dict into the appropriate xarray schema type.

    :param serialized_schema: dict representation of the schema.
    :returns: DataArraySchema or DatasetSchema.
    """
    schema_type = serialized_schema.get("schema_type")
    if schema_type == "data_array":
        return deserialize_data_array_schema(serialized_schema)
    elif schema_type == "dataset":
        return deserialize_dataset_schema(serialized_schema)
    else:
        raise ValueError(
            f"Unknown schema_type '{schema_type}'. "
            "Expected 'data_array' or 'dataset'."
        )


# ---------------------------------------------------------------------------
# YAML / JSON public helpers
# ---------------------------------------------------------------------------


def to_yaml(schema, stream=None, *, minimal: bool = True):
    """Write a DataArraySchema or DatasetSchema to yaml.

    :param schema: schema to write.
    :param stream: file stream to write to. If None, dumps to string.
    :param minimal: passed to :func:`serialize_schema`.
    :returns: yaml string if stream is None, otherwise None.
    """
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(MISSING_PYYAML_MESSAGE) from exc

    serialized = serialize_schema(schema, minimal=minimal)

    def _write_yaml(obj, stream):
        return yaml.safe_dump(obj, stream=stream, sort_keys=False)

    try:
        with Path(stream).open("w", encoding="utf-8") as f:  # type: ignore[arg-type]
            _write_yaml(serialized, f)
    except (TypeError, OSError):
        return _write_yaml(serialized, stream)


def from_yaml(yaml_schema):
    """Create a DataArraySchema or DatasetSchema from yaml.

    :param yaml_schema: str or Path to yaml file, or yaml string.
    :returns: DataArraySchema or DatasetSchema.
    """
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(MISSING_PYYAML_MESSAGE) from exc

    try:
        with Path(yaml_schema).open("r", encoding="utf-8") as f:
            serialized_schema = yaml.safe_load(f)
    except (TypeError, OSError):
        serialized_schema = yaml.safe_load(yaml_schema)

    return deserialize_schema(serialized_schema)


def to_json(schema, target=None, *, minimal: bool = True, **kwargs):
    """Write a DataArraySchema or DatasetSchema to json.

    :param schema: schema to write.
    :param target: file path or stream. If None, returns json string.
    :param minimal: passed to :func:`serialize_schema`.
    :param kwargs: passed to :func:`json.dump`.
    :returns: json string if target is None, otherwise None.
    """
    serialized = serialize_schema(schema, minimal=minimal)

    if target is None:
        return json.dumps(serialized, sort_keys=False, **kwargs)

    if isinstance(target, (str, Path)):
        with Path(target).open("w", encoding="utf-8") as f:
            json.dump(serialized, fp=f, sort_keys=False, **kwargs)
    else:
        json.dump(serialized, fp=target, sort_keys=False, **kwargs)


def from_json(source):
    """Create a DataArraySchema or DatasetSchema from json.

    :param source: str/Path to json file, json string, or readable stream.
    :returns: DataArraySchema or DatasetSchema.
    """
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
