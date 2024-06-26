"""Ibis engine and data types."""

import dataclasses
import inspect
import warnings
from typing import Any, Iterable, Optional, Union

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.types as ir
import numpy as np

from pandera import dtypes
from pandera.dtypes import immutable
from pandera.engines import engine, numpy_engine


@immutable(init=True)
class DataType(dtypes.DataType):
    """Base `DataType` for boxing Ibis data types."""

    type: Any = dataclasses.field(repr=False, init=False)
    """Native Ibis dtype boxed by the data type."""

    def __init__(self, dtype: Any):
        super().__init__()
        object.__setattr__(self, "type", ibis.dtype(dtype))
        dtype_cls = dtype if inspect.isclass(dtype) else dtype.__class__
        warnings.warn(
            f"'{dtype_cls}' support is not guaranteed.\n"
            + "Usage Tip: Consider writing a custom "
            + "pandera.dtypes.DataType or opening an issue at "
            + "https://github.com/pandera-dev/pandera"
        )

    def __post_init__(self):
        # this method isn't called if __init__ is defined
        object.__setattr__(
            self, "type", ibis.dtype(self.type)
        )  # pragma: no cover

    def check(
        self,
        pandera_dtype: dtypes.DataType,
        data_container: Optional[ir.Table] = None,
    ) -> Union[bool, Iterable[bool]]:
        try:
            return self.type == pandera_dtype.type
        except TypeError:
            return False


class Engine(
    metaclass=engine.Engine,
    base_pandera_dtypes=(DataType, numpy_engine.DataType),
):
    """Ibis data type engine."""

    @classmethod
    def dtype(cls, data_type: Any) -> dtypes.DataType:
        """Convert input into a pandas-compatible
        Pandera :class:`~pandera.dtypes.DataType` object."""
        try:
            return engine.Engine.dtype(cls, data_type)
        except TypeError:
            np_dtype = data_type().to_numpy()

        return engine.Engine.dtype(cls, np_dtype)


###############################################################################
# signed integer
###############################################################################


@Engine.register_dtype(
    equivalents=[np.int32, dtypes.Int32, dtypes.Int32(), dt.Int32, dt.int32]
)
@immutable
class Int32(DataType, dtypes.Int32):
    """Semantic representation of a :class:`dt.Int32`."""


@Engine.register_dtype(
    equivalents=[
        int,
        np.int64,
        dtypes.Int64,
        dtypes.Int64(),
        dt.Int64,
        dt.int64,
    ]
)
@immutable
class Int64(DataType, dtypes.Int64):
    """Semantic representation of a :class:`dt.Int64`."""

    type = dt.int64


###############################################################################
# float
###############################################################################


@Engine.register_dtype(
    equivalents=[
        float,
        np.float64,
        dtypes.Float64,
        dtypes.Float64(),
        dt.Float64,
        dt.float64,
    ]
)
@immutable
class Float64(DataType, dtypes.Float64):
    """Semantic representation of a :class:`dt.Float64`."""

    type = dt.float64
