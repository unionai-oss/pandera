"""Public xarray schema API (see also :mod:`pandera.xarray`)."""

from pandera.api.xarray.base import (
    BaseDataArraySchema,
    BaseDatasetSchema,
    BaseDataTreeSchema,
)
from pandera.api.xarray.components import Coordinate, DataVar
from pandera.api.xarray.container import DataArraySchema, DatasetSchema
from pandera.api.xarray.types import (
    XARRAY_CHECK_OBJECT_TYPES,
    XarrayCheckObjects,
    XarrayData,
)
from pandera.api.xarray.utils import get_validation_depth

__all__ = [
    "BaseDataArraySchema",
    "BaseDataTreeSchema",
    "BaseDatasetSchema",
    "Coordinate",
    "DataArraySchema",
    "DataVar",
    "DatasetSchema",
    "get_validation_depth",
    "XarrayData",
    "XARRAY_CHECK_OBJECT_TYPES",
    "XarrayCheckObjects",
]
