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
from pandera.api.parsers import Parser
from pandera.api.xarray import (
    Coordinate,
    DataArrayModel,
    DataArraySchema,
    DatasetModel,
    DatasetSchema,
    DataTreeModel,
    DataTreeSchema,
    DataVar,
    Field,
    XarrayData,
    get_validation_depth,
)
from pandera.backends.xarray.register import register_xarray_backends
from pandera.decorators import check_input, check_io, check_output, check_types
from pandera.io.xarray_io import from_json, from_yaml, to_json, to_yaml
from pandera.schema_inference.xarray import infer_schema

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
    "DataTreeModel",
    "DataTreeSchema",
    "DataVar",
    "Field",
    "errors",
    "from_json",
    "from_yaml",
    "get_validation_depth",
    "infer_schema",
    "Parser",
    "to_json",
    "to_yaml",
    "XarrayData",
]
