"""Module for inferring xarray DataArray/Dataset schemas from data."""

from __future__ import annotations

from typing import overload

import xarray as xr

from pandera.api.xarray.components import Coordinate, DataVar
from pandera.api.xarray.container import DataArraySchema, DatasetSchema
from pandera.schema_statistics.xarray import (
    infer_data_array_statistics,
    infer_dataset_statistics,
    parse_check_statistics,
)


@overload
def infer_schema(
    xarray_obj: xr.Dataset,
) -> DatasetSchema: ...


@overload
def infer_schema(  # type: ignore[overload-cannot-match]
    xarray_obj: xr.DataArray,
) -> DataArraySchema: ...


def infer_schema(xarray_obj):
    """Infer schema for an xarray DataArray or Dataset.

    :param xarray_obj: DataArray or Dataset to infer.
    :returns: DataArraySchema or DatasetSchema
    :raises TypeError: if xarray_obj is not a recognized type.
    """
    if isinstance(xarray_obj, xr.DataArray):
        return infer_data_array_schema(xarray_obj)
    elif isinstance(xarray_obj, xr.Dataset):
        return infer_dataset_schema(xarray_obj)
    else:
        raise TypeError(
            "xarray_obj type not recognized. Expected an xarray DataArray "
            f"or Dataset, found {type(xarray_obj)}"
        )


def _create_coordinate(coord_stats: dict) -> Coordinate:
    """Create a Coordinate from inferred statistics."""
    return Coordinate(
        dtype=coord_stats.get("dtype"),
        checks=parse_check_statistics(coord_stats.get("checks")),
        nullable=coord_stats.get("nullable", False),
    )


def infer_data_array_schema(da: xr.DataArray) -> DataArraySchema:
    """Infer a DataArraySchema from an xarray DataArray.

    :param da: DataArray to infer.
    :returns: DataArraySchema
    """
    stats = infer_data_array_statistics(da)

    coords = None
    if stats["coords"]:
        coords = {
            name: _create_coordinate(cstats)
            for name, cstats in stats["coords"].items()
        }

    return DataArraySchema(
        dtype=stats["dtype"],
        dims=stats["dims"],
        coords=coords,
        name=stats["name"],
        checks=parse_check_statistics(stats["checks"]),
        nullable=stats["nullable"],
        coerce=True,
    )


def infer_dataset_schema(ds: xr.Dataset) -> DatasetSchema:
    """Infer a DatasetSchema from an xarray Dataset.

    :param ds: Dataset to infer.
    :returns: DatasetSchema
    """
    stats = infer_dataset_statistics(ds)

    data_vars: dict[str, DataVar | DataArraySchema | None] | None = None
    if stats["data_vars"]:
        data_vars = {}
        for var_name, var_stats in stats["data_vars"].items():
            data_vars[var_name] = DataVar(
                dtype=var_stats["dtype"],
                dims=var_stats["dims"],
                checks=parse_check_statistics(var_stats.get("checks")),
                nullable=var_stats.get("nullable", False),
            )

    coords = None
    if stats["coords"]:
        coords = {
            name: _create_coordinate(cstats)
            for name, cstats in stats["coords"].items()
        }

    return DatasetSchema(
        data_vars=data_vars,
        coords=coords,
        dims=tuple(stats["dims"].keys()) if stats["dims"] else None,
        sizes=stats["dims"] if stats["dims"] else None,
    )
