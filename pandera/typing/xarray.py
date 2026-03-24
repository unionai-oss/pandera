"""Pandera type annotations for xarray."""

from __future__ import annotations

from typing import Generic, TypeVar

import xarray as xr


T = TypeVar("T")


class DataArray(xr.DataArray, Generic[T]):
    """Annotation-only generic for :class:`xarray.DataArray`."""

class Dataset(xr.Dataset, Generic[T]):
    """Annotation-only generic for :class:`xarray.Dataset`."""


__all__ = ["DataArray", "Dataset"]
