"""Pandera type annotations for xarray."""

from __future__ import annotations

from typing import Generic, TypeVar

import xarray as xr

T = TypeVar("T")
TCoord = TypeVar("TCoord")


class Coordinate(Generic[TCoord]):
    """Mark a field on :class:`~pandera.api.xarray.model.DataArrayModel` or
    :class:`~pandera.api.xarray.model.DatasetModel` as a coordinate schema.

    Usage: ``time: Coordinate[datetime64]`` or ``lat: Coordinate[float]``.
    """


class XarrayAnnotationBase:
    """Marker base for pandera xarray annotation types.

    Used by :attr:`~pandera.typing.common.AnnotationInfo.is_generic_xarray`
    to recognise ``DataArray[Model]`` and ``Dataset[Model]`` annotations in
    decorator-based validation (e.g. :func:`~pandera.decorators.check_types`).
    """


class DataArray(XarrayAnnotationBase, xr.DataArray, Generic[T]):
    """Annotation-only generic for :class:`xarray.DataArray`."""


class Dataset(XarrayAnnotationBase, xr.Dataset, Generic[T]):
    """Annotation-only generic for :class:`xarray.Dataset`."""


__all__ = [
    "Coordinate",
    "DataArray",
    "Dataset",
    "XarrayAnnotationBase",
]
