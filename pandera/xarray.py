"""Pandera validation API for xarray.

Import this module (or ``pandera.api.xarray`` types) to register xarray
backends. Typical entry points:

- :class:`~pandera.api.xarray.container.DataArraySchema`
- :class:`~pandera.api.xarray.container.DatasetSchema`
- :class:`~pandera.api.checks.Check` (same class as other backends; builtins
  dispatch on DataArray / Dataset type)
"""

from pandera import errors
from pandera.api.checks import Check
from pandera.api.xarray import (
    Coordinate,
    DataArrayModel,
    DataArraySchema,
    DatasetModel,
    DatasetSchema,
    DataVar,
    Field,
    XarrayData,
    get_validation_depth,
)
from pandera.backends.xarray.register import register_xarray_backends
from pandera.decorators import check_input, check_io, check_output, check_types

register_xarray_backends()

__all__ = [
    "check_input",
    "check_io",
    "check_output",
    "check_types",
    "Check",
    "Coordinate",
    "DataArrayModel",
    "DataArraySchema",
    "DatasetModel",
    "DatasetSchema",
    "DataVar",
    "Field",
    "errors",
    "get_validation_depth",
    "XarrayData",
]
