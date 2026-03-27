"""Xarray dtype system."""

from typing import Any

import numpy as np

from pandera.engines import engine
from pandera.engines.numpy_engine import DataType as NumpyDataType

import xarray as xr


class DataType(NumpyDataType):
    """xarray-compatible data type.

    Delegates to numpy_engine for dtype resolution since xarray arrays
    are typically backed by NumPy or duck arrays with NumPy-like dtypes.
    """

    def __init__(self, dtype: Any):
        super(NumpyDataType, self).__init__()
        object.__setattr__(self, "type", np.dtype(dtype))

    def coerce(self, data_container: "xr.DataArray") -> "xr.DataArray":
        """Coerce a DataArray to the specified dtype."""
        return data_container.astype(self.type)

    def check(
        self, pandera_dtype, data_container: "xr.DataArray | None" = None
    ) -> bool:
        """Check if the DataArray dtype matches."""
        if data_container is None:
            return False
        return np.issubdtype(data_container.dtype, self.type)


class Engine(metaclass=engine.Engine, base_pandera_dtypes=DataType):
    """xarray data type engine."""

    @classmethod
    def dtype(cls, data_type: Any) -> "DataType":
        """Convert input into an xarray-compatible DataType."""
        try:
            return engine.Engine.dtype(cls, data_type)  # type: ignore
        except TypeError:
            return cls._dtype_from_numpy_type(data_type)

    @classmethod
    def _dtype_from_numpy_type(cls, data_type: Any) -> "DataType":
        """Create a DataType from a numpy-like type."""
        return DataType(data_type)
