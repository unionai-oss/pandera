"""Abstract base classes for xarray schema APIs."""

from pandera.api.base.schema import BaseSchema


class BaseDataArraySchema(BaseSchema):
    """Base class for xarray :class:`~xarray.DataArray` schemas."""


class BaseDatasetSchema(BaseSchema):
    """Base class for xarray :class:`~xarray.Dataset` schemas."""


class BaseDataTreeSchema(BaseSchema):
    """Base class for xarray :class:`~xarray.DataTree` schemas."""
