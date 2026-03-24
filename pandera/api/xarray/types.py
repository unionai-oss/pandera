"""Xarray type definitions."""

from typing import NamedTuple, Union

import xarray as xr


class XarrayData(NamedTuple):
    """Container passed to Check backends."""

    obj: Union["xr.DataArray", "xr.Dataset"]
    key: str | None = None


XARRAY_CHECK_OBJECT_TYPES: tuple[type, ...] = (xr.DataArray, xr.Dataset)

# Backwards-compatible alias (tuple of concrete types, not a Union).
XarrayCheckObjects = XARRAY_CHECK_OBJECT_TYPES
