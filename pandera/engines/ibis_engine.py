"""Ibis engine and data types."""

import dataclasses
import datetime
import inspect
import warnings
from typing import Any, Literal, Optional, Union
from collections.abc import Iterable

import ibis
import ibis.expr.datatypes as dt
import numpy as np
from ibis.common.temporal import IntervalUnit

from pandera import dtypes
from pandera.dtypes import immutable
from pandera.engines import engine, numpy_engine

IbisObject = Union[ibis.Column, ibis.Table]


@immutable(init=True)
class DataType(dtypes.DataType):
    """Base `DataType` for boxing Ibis data types."""

    type: Any = dataclasses.field(repr=False, init=False)
    """Native Ibis dtype boxed by the data type."""

    def __init__(self, dtype: Optional[Any] = None):
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
            if isinstance(data_container, ibis.Column)
            else dict.fromkeys(data_container.columns, self.type)
        )

    def check(
        self,
        pandera_dtype: dtypes.DataType,
        data_container: Optional[ibis.Table] = None,
    ) -> Union[bool, Iterable[bool]]:
        try:
            return self.type == pandera_dtype.type
        except TypeError:
            return False

    def __str__(self) -> str:
        return str(self.type)

    def __repr__(self) -> str:
        return f"DataType({self})"


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
# boolean
###############################################################################


@Engine.register_dtype(
    equivalents=[
        bool,
        np.bool_,
        dtypes.Bool,
        dtypes.Bool(),
        dt.Boolean,
        dt.boolean,
    ]
)
@immutable
class Bool(DataType, dtypes.Bool):
    """Semantic representation of a :class:`dt.Boolean`."""

    type = dt.boolean


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

    type = dt.float32


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


@Engine.register_dtype(
    equivalents=[
        bytes,
        np.bytes_,
        dtypes.Binary,
        dtypes.Binary(),
        dt.Binary,
        dt.binary,
    ]
)
@immutable
class Binary(DataType, dtypes.Binary):
    """Semantic representation of a :class:`dt.Binary`."""

    type = dt.binary


###############################################################################
# temporal
###############################################################################


@Engine.register_dtype(
    equivalents=[
        "date",
        datetime.date,
        dtypes.Date,
        dtypes.Date(),
        dt.Date,
        dt.date,
    ]
)
@immutable
class Date(DataType, dtypes.Date):
    """Semantic representation of a :class:`dt.Date`."""

    type = dt.date


@Engine.register_dtype(
    equivalents=[
        "datetime",
        datetime.datetime,
        np.datetime64,
        dtypes.DateTime,
        dtypes.DateTime(),
        dt.Timestamp,
        dt.timestamp,
    ]
)
@immutable(init=True)
class DateTime(DataType, dtypes.DateTime):
    """Semantic representation of a :class:`dt.Timestamp`."""

    type: type[dt.Timestamp]

    def __init__(
        self,
        timezone: Optional[str] = None,
        scale: Optional[Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]] = None,
    ):
        object.__setattr__(
            self, "type", dt.Timestamp(timezone=timezone, scale=scale)
        )

    @classmethod
    def from_parametrized_dtype(cls, ibis_dtype: dt.Timestamp):
        """Convert a :class:`dt.Timestamp` to a Pandera
        :class:`~pandera.engines.ibis_engine.DateTime`."""
        return cls(timezone=ibis_dtype.timezone, scale=ibis_dtype.scale)


@Engine.register_dtype(
    equivalents=[
        "time",
        datetime.time,
        dt.Time,
        dt.time,
    ]
)
@immutable
class Time(DataType):
    """Semantic representation of a :class:`dt.Time`."""

    type = dt.time


@Engine.register_dtype(
    equivalents=[
        "timedelta",
        datetime.timedelta,
        np.timedelta64,
        dtypes.Timedelta,
        dtypes.Timedelta(),
        dt.Interval,
    ]
)
@immutable(init=True)
class Timedelta(DataType, dtypes.DateTime):
    """Semantic representation of a :class:`dt.Timestamp`."""

    type: type[dt.Interval]

    def __init__(self, unit: IntervalUnit = "us"):
        object.__setattr__(self, "type", dt.Interval(unit=unit))

    @classmethod
    def from_parametrized_dtype(cls, ibis_dtype: dt.Interval):
        """Convert a :class:`dt.Interval` to a Pandera
        :class:`~pandera.engines.ibis_engine.Timedelta`."""
        return cls(unit=ibis_dtype.unit)
