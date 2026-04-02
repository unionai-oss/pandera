"""Module for inferring and extracting statistics from xarray objects."""

from __future__ import annotations

import warnings
from typing import Any, Union

import numpy as np
import xarray as xr

from pandera.api.checks import Check


def infer_data_array_statistics(da: xr.DataArray) -> dict[str, Any]:
    """Infer statistics from an xarray DataArray.

    :param da: DataArray to infer statistics from.
    :returns: dict with dtype, dims, nullable, name, coords, and checks.
    """
    return {
        "dtype": _get_dtype_string(da),
        "dims": da.dims,
        "nullable": bool(da.isnull().any().values),
        "name": da.name,
        "coords": _infer_coord_statistics(da),
        "checks": _get_array_check_statistics(da),
    }


def infer_dataset_statistics(ds: xr.Dataset) -> dict[str, Any]:
    """Infer statistics from an xarray Dataset.

    :param ds: Dataset to infer statistics from.
    :returns: dict with data_vars, coords, dims, and sizes.
    """
    data_var_stats = {}
    for var_name in ds.data_vars:
        da = ds[var_name]
        data_var_stats[var_name] = {
            "dtype": _get_dtype_string(da),
            "dims": da.dims,
            "nullable": bool(da.isnull().any().values),
            "checks": _get_array_check_statistics(da),
        }

    coord_stats = {}
    for coord_name in ds.coords:
        coord_da = ds.coords[coord_name]
        coord_stats[coord_name] = {
            "dtype": _get_dtype_string(coord_da),
            "nullable": bool(coord_da.isnull().any().values),
            "checks": _get_array_check_statistics(coord_da),
        }

    return {
        "data_vars": data_var_stats if data_var_stats else None,
        "coords": coord_stats if coord_stats else None,
        "dims": dict(ds.sizes),
    }


def get_data_array_schema_statistics(
    data_array_schema,
) -> dict[str, Any]:
    """Extract statistics from a DataArraySchema object.

    :param data_array_schema: the schema to extract statistics from.
    :returns: dict with structural and check statistics.
    """
    coord_stats: dict[str, Any] | None = None
    if data_array_schema.coords:
        if isinstance(data_array_schema.coords, list):
            coord_stats = {name: {} for name in data_array_schema.coords}
        else:
            coord_stats = {}
            for name, coord in data_array_schema.coords.items():
                coord_stats[name] = {
                    "dtype": (
                        str(coord.dtype) if coord.dtype is not None else None
                    ),
                    "nullable": coord.nullable,
                    "checks": parse_checks(coord.checks),
                    "title": coord.title,
                    "description": coord.description,
                }

    return {
        "dtype": (
            str(data_array_schema.dtype)
            if data_array_schema.dtype is not None
            else None
        ),
        "dims": data_array_schema.dims,
        "ordered_dims": data_array_schema.ordered_dims,
        "sizes": data_array_schema.sizes,
        "shape": data_array_schema.shape,
        "name": data_array_schema.name,
        "nullable": data_array_schema.nullable,
        "coerce": data_array_schema.coerce,
        "coords": coord_stats,
        "checks": parse_checks(data_array_schema.checks),
        "title": data_array_schema.title,
        "description": data_array_schema.description,
    }


def get_dataset_schema_statistics(dataset_schema) -> dict[str, Any]:
    """Extract statistics from a DatasetSchema object.

    :param dataset_schema: the schema to extract statistics from.
    :returns: dict with data_vars, coords, dims, and schema-level checks.
    """
    from pandera.api.xarray.components import Coordinate, DataVar

    data_var_stats: dict[str, dict[str, Any]] | None = None
    if dataset_schema.data_vars:
        data_var_stats = {}
        for key, spec in dataset_schema.data_vars.items():
            if spec is None:
                data_var_stats[key] = {"required": True}
                continue
            if isinstance(spec, DataVar):
                data_var_stats[key] = {
                    "dtype": (
                        str(spec.dtype) if spec.dtype is not None else None
                    ),
                    "dims": spec.dims,
                    "nullable": spec.nullable,
                    "coerce": spec.coerce,
                    "required": spec.required,
                    "regex": spec.regex,
                    "alias": spec.alias,
                    "checks": parse_checks(spec.checks),
                    "title": spec.title,
                    "description": spec.description,
                }
            else:
                data_var_stats[key] = get_data_array_schema_statistics(spec)

    coord_stats: dict[str, Any] | None = None
    if dataset_schema.coords:
        if isinstance(dataset_schema.coords, list):
            coord_stats = {name: {} for name in dataset_schema.coords}
        else:
            coord_stats = {}
            for name, coord in dataset_schema.coords.items():
                if isinstance(coord, Coordinate):
                    coord_stats[name] = {
                        "dtype": (
                            str(coord.dtype)
                            if coord.dtype is not None
                            else None
                        ),
                        "nullable": coord.nullable,
                        "checks": parse_checks(coord.checks),
                        "title": coord.title,
                        "description": coord.description,
                    }
                else:
                    coord_stats[name] = {}

    return {
        "data_vars": data_var_stats,
        "coords": coord_stats,
        "dims": dataset_schema.dims,
        "ordered_dims": dataset_schema.ordered_dims,
        "sizes": dataset_schema.sizes,
        "strict": dataset_schema.strict,
        "strict_coords": dataset_schema.strict_coords,
        "strict_attrs": dataset_schema.strict_attrs,
        "checks": parse_checks(dataset_schema.checks),
        "title": dataset_schema.title,
        "description": dataset_schema.description,
    }


def parse_checks(checks) -> Union[list[dict[str, Any]], None]:
    """Convert Check objects to serializable statistics.

    Follows the same format as
    :func:`pandera.schema_statistics.pandas.parse_checks`: each entry is
    a dict of check statistics with an ``"options"`` sub-dict containing
    ``check_name`` and other metadata.

    :param checks: list of Check objects.
    :returns: list of dicts with check statistics, or None.
    """
    if not checks:
        return None

    check_statistics: list[dict[str, Any]] = []

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
        check_options = {
            k: v for k, v in check_options.items() if v is not None
        }

        if check_options:
            base_stats["options"] = check_options
            check_statistics.append(base_stats)

    _warn_incompatible_checks(check_statistics)

    return check_statistics if check_statistics else None


def parse_check_statistics(
    check_stats: dict[str, Any] | None,
) -> list[Check] | None:
    """Convert check statistics dict to a list of Check objects.

    Takes the same ``{check_name: value_or_dict}`` format produced by
    :func:`_get_array_check_statistics` (inference path) and turns each
    entry into a :class:`~pandera.api.checks.Check` instance — mirroring
    :func:`pandera.schema_statistics.pandas.parse_check_statistics`.

    :param check_stats: dict mapping check names to their statistics.
    :returns: list of Check objects, or None.
    """
    if check_stats is None:
        return None

    checks: list[Check] = []
    for check_name, stats in check_stats.items():
        check_fn = getattr(Check, check_name, None)
        if check_fn is None:
            warnings.warn(
                f"Check `{check_name}` not found, skipping.",
            )
            continue
        try:
            if isinstance(stats, dict):
                options = (
                    stats.pop("options", {}) if "options" in stats else {}
                )
                if stats:
                    check_instance = check_fn(**stats)
                else:
                    check_instance = check_fn()
                for option_name, option_value in options.items():
                    setattr(check_instance, option_name, option_value)
                checks.append(check_instance)
            else:
                checks.append(check_fn(stats))
        except TypeError:
            checks.append(check_fn(stats))

    return checks if checks else None


def _warn_incompatible_checks(
    check_statistics: list[dict[str, Any]],
) -> None:
    """Warn when multiple mutually-exclusive bound checks are present."""
    incompatible_checks = {
        "equal_to": "eq",
        "greater_than": "gt",
        "less_than": "lt",
        "greater_than_or_equal_to": "ge",
        "less_than_or_equal_to": "le",
        "in_range": "between",
    }
    count = sum(
        1
        for cs in check_statistics
        if cs.get("options", {}).get("check_name") in incompatible_checks
    )
    if count > 1:
        msg = ", ".join(f"{k} ({v})" for k, v in incompatible_checks.items())
        warnings.warn(f"You do not need more than one check out of {msg}.")


def _get_dtype_string(da: xr.DataArray) -> str:
    """Get a string representation of the DataArray's dtype."""
    return str(da.dtype)


def _get_array_check_statistics(
    da: xr.DataArray,
) -> Union[dict[str, Any], None]:
    """Infer check statistics (min/max bounds) from array data.

    Returns a ``{check_name: value}`` dict matching the format used by
    :func:`pandera.schema_statistics.pandas._get_array_check_statistics`.
    """
    if da.size == 0:
        return None
    try:
        if da.isnull().all().values:
            return None
    except (TypeError, ValueError):
        pass

    dtype = da.dtype
    if np.issubdtype(dtype, np.datetime64) or np.issubdtype(
        dtype, np.timedelta64
    ):
        check_stats = {
            "greater_than_or_equal_to": da.min().values.item(),
            "less_than_or_equal_to": da.max().values.item(),
        }
    elif np.issubdtype(dtype, np.number) and not np.issubdtype(
        dtype, np.bool_
    ):
        check_stats = {
            "greater_than_or_equal_to": float(da.min().values),
            "less_than_or_equal_to": float(da.max().values),
        }
    else:
        check_stats = {}

    return check_stats if check_stats else None


def _infer_coord_statistics(da: xr.DataArray) -> Union[dict[str, Any], None]:
    """Infer coordinate statistics from a DataArray."""
    if not da.coords:
        return None
    coord_stats = {}
    for coord_name in da.coords:
        coord_da = da.coords[coord_name]
        coord_stats[coord_name] = {
            "dtype": _get_dtype_string(coord_da),
            "nullable": bool(coord_da.isnull().any().values),
            "checks": _get_array_check_statistics(coord_da),
        }
    return coord_stats if coord_stats else None
