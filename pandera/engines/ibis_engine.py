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

IbisObject = Union[ir.Column, ir.Table]


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

    def coerce(self, data_container: IbisObject) -> IbisObject:
        """Coerce data container to the data type."""
        return data_container.cast(
            self.type
            if isinstance(data_container, ir.Column)
            else dict.fromkeys(data_container.columns, self.type)
        )

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
    equivalents=[np.int8, dtypes.Int8, dtypes.Int8(), dt.Int8, dt.int8]
)
@immutable
class Int8(DataType, dtypes.Int8):
    """Semantic representation of a :class:`dt.Int8`."""

    type = dt.int8


@Engine.register_dtype(
    equivalents=[np.int16, dtypes.Int16, dtypes.Int16(), dt.Int16, dt.int16]
)
@immutable
class Int16(DataType, dtypes.Int16):
    """Semantic representation of a :class:`dt.Int16`."""

    type = dt.int16


@Engine.register_dtype(
    equivalents=[np.int32, dtypes.Int32, dtypes.Int32(), dt.Int32, dt.int32]
)
@immutable
class Int32(DataType, dtypes.Int32):
    """Semantic representation of a :class:`dt.Int32`."""

    type = dt.int32


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
# unsigned integer
###############################################################################


@Engine.register_dtype(
    equivalents=[np.uint8, dtypes.UInt8, dtypes.UInt8(), dt.UInt8, dt.uint8]
)
@immutable
class UInt8(DataType, dtypes.UInt8):
    """Semantic representation of a :class:`dt.UInt8`."""

    type = dt.uint8


@Engine.register_dtype(
    equivalents=[
        np.uint16,
        dtypes.UInt16,
        dtypes.UInt16(),
        dt.UInt16,
        dt.uint16,
    ]
)
@immutable
class UInt16(DataType, dtypes.UInt16):
    """Semantic representation of a :class:`dt.UInt16`."""

    type = dt.uint16


@Engine.register_dtype(
    equivalents=[
        np.uint32,
        dtypes.UInt32,
        dtypes.UInt32(),
        dt.UInt32,
        dt.uint32,
    ]
)
@immutable
class UInt32(DataType, dtypes.UInt32):
    """Semantic representation of a :class:`dt.UInt32`."""

    type = dt.uint32


@Engine.register_dtype(
    equivalents=[
        np.uint64,
        dtypes.UInt64,
        dtypes.UInt64(),
        dt.UInt64,
        dt.uint64,
    ]
)
@immutable
class UInt64(DataType, dtypes.UInt64):
    """Semantic representation of a :class:`dt.UInt64`."""

    type = dt.uint64


###############################################################################
# float
###############################################################################


@Engine.register_dtype(
    equivalents=[
        np.float32,
        dtypes.Float32,
        dtypes.Float32(),
        dt.Float32,
        dt.float32,
    ]
)
@immutable
class Float32(DataType, dtypes.Float32):
    """Semantic representation of a :class:`dt.Float32`."""


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


###############################################################################
# nominal
###############################################################################


@Engine.register_dtype(
    equivalents=[
        str,
        np.str_,
        dtypes.String,
        dtypes.String(),
        dt.String,
        dt.string,
    ]
)
@immutable
class String(DataType, dtypes.String):
    """Semantic representation of a :class:`dt.String`."""

    type = dt.string
