"""Narwhals engine and data types."""

import dataclasses
from typing import Any, Optional, Union

import narwhals.stable.v1 as nw

from pandera import dtypes, errors
from pandera.api.narwhals.types import NarwhalsData
from pandera.api.narwhals.utils import _materialize, _to_native
from pandera.dtypes import immutable
from pandera.engines import engine

NarwhalsDataContainer = Any  # Union[nw.LazyFrame, NarwhalsData]

COERCION_ERRORS = (
    TypeError,
    nw.exceptions.InvalidOperationError,
    nw.exceptions.ComputeError,
)


@immutable(init=True)
class DataType(dtypes.DataType):
    """Base ``DataType`` for boxing Narwhals data types."""

    type: Any = dataclasses.field(repr=False, init=False)

    def coerce(self, data_container: NarwhalsDataContainer) -> nw.LazyFrame:
        """Coerce data container to the data type.

        Accepts a :class:`~pandera.api.narwhals.types.NarwhalsData` or a
        ``nw.LazyFrame``.  Always returns a ``nw.LazyFrame`` (lazy — does
        not collect).
        """
        if isinstance(data_container, nw.LazyFrame):
            data_container = NarwhalsData(frame=data_container)

        lf: nw.LazyFrame = data_container.frame
        key: str = data_container.key

        if key == "*":
            return lf.with_columns(nw.all().cast(self.type))
        return lf.with_columns(nw.col(key).cast(self.type))

    def try_coerce(
        self, data_container: NarwhalsDataContainer
    ) -> nw.LazyFrame:
        """Coerce data container to the data type.

        Raises a :class:`~pandera.errors.ParserError` if the coercion fails.

        :raises: :class:`~pandera.errors.ParserError`: if coercion fails
        """
        if isinstance(data_container, nw.LazyFrame):
            data_container = NarwhalsData(frame=data_container)

        try:
            lf = self.coerce(data_container)
            # Bounded probe: exercise the cast with 1 row instead of full frame.
            # For nw.LazyFrame (polars): head(1).collect() stays in Narwhals.
            # For nw.DataFrame (ibis): _materialize(head(1)) handles .execute().
            if isinstance(lf, nw.LazyFrame):
                lf.head(1).collect()
            else:
                _materialize(lf.head(1))
            return lf
        except COERCION_ERRORS as exc:
            key = data_container.key
            _key = "" if key == "*" else f"'{key}' in"
            # Produce native failure_cases: use _materialize to handle both polars
            # (nw.LazyFrame -> collect) and ibis (nw.DataFrame -> .execute()) backends.
            failure_cases = _to_native(_materialize(data_container.frame))
            if key != "*":
                try:
                    failure_cases = failure_cases.select(key)
                except Exception:
                    pass
            raise errors.ParserError(
                f"Could not coerce {_key} LazyFrame "
                f"into type {self.type}",
                failure_cases=failure_cases,
            ) from exc

    def __str__(self) -> str:
        return str(self.type)

    def __repr__(self) -> str:
        return f"DataType({self})"


class Engine(metaclass=engine.Engine, base_pandera_dtypes=DataType):
    """Narwhals data type engine."""

    @classmethod
    def dtype(cls, data_type: Any) -> dtypes.DataType:
        """Convert input into a Narwhals-compatible
        Pandera :class:`~pandera.dtypes.DataType` object.

        If ``data_type`` is an engine-specific dtype from another backend
        (e.g. ``polars_engine.Int64``), the method falls back to the shared
        abstract pandera base class (e.g. ``dtypes.Int64``) so cross-engine
        dtype comparisons work without importing the foreign engine.
        """
        try:
            return engine.Engine.dtype(cls, data_type)
        except TypeError:
            # data_type may be an instance from another engine (polars, ibis, …).
            # Each engine-specific class inherits from a shared abstract pandera
            # dtype (e.g. polars_engine.Int64 → dtypes.Int64). Re-interpreting
            # through that abstract base lets narwhals_engine handle cross-engine
            # dtype inputs. Parametric types (List, Struct) have no abstract base
            # in pandera.dtypes and will still raise TypeError.
            bases = type(data_type).__bases__
            abstract_base = bases[-1] if bases else None
            if abstract_base is not None and abstract_base is not dtypes.DataType:
                try:
                    return engine.Engine.dtype(cls, abstract_base())
                except TypeError:
                    pass
            raise TypeError(
                f"data type '{data_type}' not understood by "
                f"{cls.__name__}."
            ) from None


###############################################################################
# Integer types
###############################################################################

@Engine.register_dtype(
    equivalents=["int8", nw.Int8, dtypes.Int8, dtypes.Int8()]
)
@immutable
class Int8(DataType, dtypes.Int8):
    """Narwhals signed 8-bit integer data type."""

    type = nw.Int8


@Engine.register_dtype(
    equivalents=["int16", nw.Int16, dtypes.Int16, dtypes.Int16()]
)
@immutable
class Int16(DataType, dtypes.Int16):
    """Narwhals signed 16-bit integer data type."""

    type = nw.Int16


@Engine.register_dtype(
    equivalents=["int32", nw.Int32, dtypes.Int32, dtypes.Int32()]
)
@immutable
class Int32(DataType, dtypes.Int32):
    """Narwhals signed 32-bit integer data type."""

    type = nw.Int32


@Engine.register_dtype(
    equivalents=["int64", nw.Int64, dtypes.Int64, dtypes.Int64()]
)
@immutable
class Int64(DataType, dtypes.Int64):
    """Narwhals signed 64-bit integer data type."""

    type = nw.Int64


@Engine.register_dtype(
    equivalents=["uint8", nw.UInt8, dtypes.UInt8, dtypes.UInt8()]
)
@immutable
class UInt8(DataType, dtypes.UInt8):
    """Narwhals unsigned 8-bit integer data type."""

    type = nw.UInt8


@Engine.register_dtype(
    equivalents=["uint16", nw.UInt16, dtypes.UInt16, dtypes.UInt16()]
)
@immutable
class UInt16(DataType, dtypes.UInt16):
    """Narwhals unsigned 16-bit integer data type."""

    type = nw.UInt16


@Engine.register_dtype(
    equivalents=["uint32", nw.UInt32, dtypes.UInt32, dtypes.UInt32()]
)
@immutable
class UInt32(DataType, dtypes.UInt32):
    """Narwhals unsigned 32-bit integer data type."""

    type = nw.UInt32


@Engine.register_dtype(
    equivalents=["uint64", nw.UInt64, dtypes.UInt64, dtypes.UInt64()]
)
@immutable
class UInt64(DataType, dtypes.UInt64):
    """Narwhals unsigned 64-bit integer data type."""

    type = nw.UInt64


###############################################################################
# Floating-point types
###############################################################################

@Engine.register_dtype(
    equivalents=["float32", nw.Float32, dtypes.Float32, dtypes.Float32()]
)
@immutable
class Float32(DataType, dtypes.Float32):
    """Narwhals 32-bit floating point data type."""

    type = nw.Float32


@Engine.register_dtype(
    equivalents=["float64", nw.Float64, dtypes.Float64, dtypes.Float64()]
)
@immutable
class Float64(DataType, dtypes.Float64):
    """Narwhals 64-bit floating point data type."""

    type = nw.Float64


###############################################################################
# String / Boolean types
###############################################################################

@Engine.register_dtype(
    equivalents=["str", "string", nw.String, dtypes.String, dtypes.String()]
)
@immutable
class String(DataType, dtypes.String):
    """Narwhals string data type."""

    type = nw.String


@Engine.register_dtype(
    equivalents=["bool", nw.Boolean, dtypes.Bool, dtypes.Bool()]
)
@immutable
class Bool(DataType, dtypes.Bool):
    """Narwhals boolean data type."""

    type = nw.Boolean


###############################################################################
# Temporal types
###############################################################################

@Engine.register_dtype(
    equivalents=["date", nw.Date, dtypes.Date, dtypes.Date()]
)
@immutable
class Date(DataType, dtypes.Date):
    """Narwhals date data type."""

    type = nw.Date


@Engine.register_dtype(
    equivalents=["datetime", nw.Datetime, dtypes.DateTime, dtypes.DateTime()]
)
@immutable(init=True)
class DateTime(DataType, dtypes.DateTime):
    """Narwhals datetime data type."""

    type: Any = nw.Datetime

    def __init__(
        self,
        time_unit: Optional[str] = None,
        time_zone: Optional[str] = None,
    ) -> None:
        if time_unit is not None:
            object.__setattr__(
                self, "type", nw.Datetime(time_unit, time_zone)
            )
        else:
            object.__setattr__(self, "type", nw.Datetime)

    @classmethod
    def from_parametrized_dtype(cls, nw_dtype: Any) -> Any:
        """Convert a parameterized ``nw.Datetime`` instance to a Pandera
        :class:`DateTime`."""
        return cls(
            time_unit=nw_dtype.time_unit,
            time_zone=nw_dtype.time_zone,
        )


@Engine.register_dtype(
    equivalents=[
        "duration",
        nw.Duration,
        dtypes.Timedelta,
        dtypes.Timedelta(),
    ]
)
@immutable(init=True)
class Duration(DataType, dtypes.Timedelta):
    """Narwhals duration data type."""

    type: Any = nw.Duration

    def __init__(
        self,
        time_unit: Optional[str] = None,
    ) -> None:
        if time_unit is not None:
            object.__setattr__(self, "type", nw.Duration(time_unit))
        else:
            object.__setattr__(self, "type", nw.Duration)

    @classmethod
    def from_parametrized_dtype(cls, nw_dtype: Any) -> Any:
        """Convert a parameterized ``nw.Duration`` instance to a Pandera
        :class:`Duration`."""
        return cls(time_unit=nw_dtype.time_unit)


###############################################################################
# Nested types
###############################################################################

@Engine.register_dtype(
    equivalents=[
        "category",
        "categorical",
        nw.Categorical,
        dtypes.Category,
        dtypes.Category(),
    ]
)
@immutable
class Categorical(DataType, dtypes.Category):
    """Narwhals categorical data type."""

    type = nw.Categorical


@Engine.register_dtype(equivalents=["list", nw.List])
@immutable(init=True)
class List(DataType):
    """Narwhals List nested type."""

    type: Any = nw.List

    def __init__(self, inner: Any = None) -> None:
        if inner is not None:
            object.__setattr__(self, "type", nw.List(inner))

    @classmethod
    def from_parametrized_dtype(cls, nw_dtype: Any) -> Any:
        """Convert a parameterized ``nw.List`` instance to a Pandera
        :class:`List`."""
        return cls(inner=nw_dtype.inner)


@Engine.register_dtype(equivalents=["struct", nw.Struct])
@immutable
class Struct(DataType):
    """Narwhals Struct nested type."""

    type = nw.Struct
