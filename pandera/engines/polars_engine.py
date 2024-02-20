"""Polars engine and data types."""
import dataclasses
import datetime
import decimal
import inspect
import warnings
from typing import Any, Union, Optional, Iterable, Literal


import polars as pl
from polars.datatypes import py_type_to_dtype

from pandera import dtypes, errors
from pandera.dtypes import immutable
from pandera.engines import engine
from pandera.engines.type_aliases import PolarsObject
from pandera.engines.utils import (
    polars_coerce_failure_cases,
    polars_object_coercible,
    polars_failure_cases_from_coercible,
    check_polars_container_all_true,
)


@immutable(init=True)
class DataType(dtypes.DataType):
    """Base `DataType` for boxing Polars data types."""

    type: pl.DataType = dataclasses.field(repr=False, init=False)

    def __init__(self, dtype: Optional[Any] = None):
        super().__init__()
        try:
            object.__setattr__(self, "type", py_type_to_dtype(dtype))
        except ValueError:
            object.__setattr__(self, "type", pl.Object)

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
            self, "type", py_type_to_dtype(self.type)
        )  # pragma: no cover

    def coerce(self, data_container: PolarsObject) -> PolarsObject:
        """Coerce data container to the data type."""
        return data_container.cast(self.type, strict=True)

    def try_coerce(self, data_container: PolarsObject) -> PolarsObject:
        """Coerce data container to the data type,
        raises a :class:`~pandera.errors.ParserError` if the coercion fails
        :raises: :class:`~pandera.errors.ParserError`: if coercion fails
        """
        try:
            return self.coerce(data_container)
        except Exception as exc:  # pylint:disable=broad-except
            raise errors.ParserError(
                f"Could not coerce {type(data_container)} data_container "
                f"into type {self.type}",
                failure_cases=polars_coerce_failure_cases(
                    data_container=data_container, type_=self.type
                ),
            ) from exc

    def check(
        self,
        pandera_dtype: dtypes.DataType,
        data_container: Optional[PolarsObject] = None,
    ) -> Union[bool, Iterable[bool]]:
        try:
            pandera_dtype = Engine.dtype(pandera_dtype)
        except TypeError:
            return False

        return self.type == pandera_dtype.type and super().check(pandera_dtype)

    def __str__(self) -> str:
        return str(self.type)

    def __repr__(self) -> str:
        return f"DataType({self})"


class Engine(  # pylint:disable=too-few-public-methods
    metaclass=engine.Engine, base_pandera_dtypes=DataType
):
    """Polars data type engine."""

    @classmethod
    def dtype(cls, data_type: Any) -> dtypes.DataType:
        """Convert input into a polars-compatible
        Pandera :class:`~pandera.dtypes.DataType` object."""
        try:
            return engine.Engine.dtype(cls, data_type)
        except TypeError:
            try:
                pl_dtype = py_type_to_dtype(data_type)
            except ValueError:
                raise TypeError(
                    f"data type '{data_type}' not understood by "
                    f"{cls.__name__}."
                ) from None

            try:
                return engine.Engine.dtype(cls, pl_dtype)
            except TypeError:
                return DataType(data_type)


###############################################################################
# Numeric types
###############################################################################
@Engine.register_dtype(
    equivalents=["int8", pl.Int8, dtypes.Int8, dtypes.Int8()]
)
@immutable
class Int8(DataType, dtypes.Int8):
    """Polars signed 8-bit integer data type."""

    type = pl.Int8


@Engine.register_dtype(
    equivalents=["int16", pl.Int16, dtypes.Int16, dtypes.Int16()]
)
@immutable
class Int16(DataType, dtypes.Int16):
    """Polars signed 16-bit integer data type."""

    type = pl.Int16


@Engine.register_dtype(
    equivalents=["int32", pl.Int32, dtypes.Int32, dtypes.Int32()]
)
@immutable
class Int32(DataType, dtypes.Int32):
    """Polars signed 32-bit integer data type."""

    type = pl.Int32


@Engine.register_dtype(
    equivalents=["int64", int, pl.Int64, dtypes.Int64, dtypes.Int64()]
)
@immutable
class Int64(DataType, dtypes.Int64):
    """Polars signed 64-bit integer data type."""

    type = pl.Int64


@Engine.register_dtype(
    equivalents=["uint8", pl.UInt8, dtypes.UInt8, dtypes.UInt8()]
)
@immutable
class UInt8(DataType, dtypes.UInt8):
    """Polars unsigned 8-bit integer data type."""

    type = pl.UInt8


@Engine.register_dtype(
    equivalents=["uint16", pl.UInt16, dtypes.UInt16, dtypes.UInt16()]
)
@immutable
class UInt16(DataType, dtypes.UInt16):
    """Polars unsigned 16-bit integer data type."""

    type = pl.UInt16


@Engine.register_dtype(
    equivalents=["uint32", pl.UInt32, dtypes.UInt32, dtypes.UInt32()]
)
@immutable
class UInt32(DataType, dtypes.UInt32):
    """Polars unsigned 32-bit integer data type."""

    type = pl.UInt32


@Engine.register_dtype(
    equivalents=["uint64", pl.UInt64, dtypes.UInt64, dtypes.UInt64()]
)
@immutable
class UInt64(DataType, dtypes.UInt64):
    """Polars unsigned 64-bit integer data type."""

    type = pl.UInt64


@Engine.register_dtype(
    equivalents=["float32", pl.Float32, dtypes.Float32, dtypes.Float32()]
)
@immutable
class Float32(DataType, dtypes.Float32):
    """Polars 32-bit floating point data type."""

    type = pl.Float32


@Engine.register_dtype(
    equivalents=[
        "float64",
        float,
        pl.Float64,
        dtypes.Float64,
        dtypes.Float64(),
    ]
)
@immutable
class Float64(DataType, dtypes.Float64):
    """Polars 64-bit floating point data type."""

    type = pl.Float64


@Engine.register_dtype(
    equivalents=[
        "decimal",
        decimal.Decimal,
        pl.Decimal,
        dtypes.Decimal,
        dtypes.Decimal(),
    ]
)
@immutable(init=True)
class Decimal(DataType, dtypes.Decimal):
    """Polars decimal data type."""

    type = pl.Float64

    def __init__(  # pylint:disable=super-init-not-called
        self,
        precision: int = dtypes.DEFAULT_PYTHON_PREC,
        scale: int = 0,
    ) -> None:
        dtypes.Decimal.__init__(
            self, precision=precision, scale=scale, rounding=None
        )

    @classmethod
    def from_parametrized_dtype(cls, polars_dtype: pl.Decimal):
        """Convert a :class:`polars.Decimal` to
        a Pandera :class:`pandera.engines.polars_engine.Decimal`."""
        return cls(precision=polars_dtype.precision, scale=polars_dtype.scale)

    def coerce(self, data_container: PolarsObject) -> PolarsObject:
        """Coerce data container to the data type."""
        data_container = data_container.cast(pl.Float64)
        return data_container.cast(
            pl.Decimal(scale=self.scale, precision=self.precision), strict=True
        )

    def check(
        self,
        pandera_dtype: dtypes.DataType,
        data_container: Any = None,  # pylint: disable=unused-argument)
    ) -> Union[bool, Iterable[bool]]:
        try:
            pandera_dtype = Engine.dtype(pandera_dtype)
            assert isinstance(
                pandera_dtype, Decimal
            ), "The return is expected to be of Decimal class"
        except TypeError:  # pragma: no cover
            return False

        try:
            return (
                (self.type == pandera_dtype.type)
                & (self.scale == pandera_dtype.scale)
                & (self.precision == pandera_dtype.precision)
            )

        except TypeError:  # pragma: no cover
            return super().check(pandera_dtype)

    def __str__(self) -> str:
        return f"Decimal(precision={self.precision}, scale={self.scale})"


###############################################################################
# Temporal types
###############################################################################


@Engine.register_dtype(
    equivalents=[
        "date",
        datetime.date,
        pl.Date,
        dtypes.Date,
        dtypes.Date(),
    ]
)
@immutable
class Date(DataType, dtypes.Date):
    """Polars date data type."""

    type = pl.Date


@Engine.register_dtype(
    equivalents=[
        "datetime",
        datetime.datetime,
        pl.Datetime,
        dtypes.DateTime,
        dtypes.DateTime(),
    ]
)
@immutable(init=True)
class DateTime(DataType, dtypes.DateTime):
    """Polars datetime data type."""

    type = pl.Datetime

    def __init__(  # pylint:disable=super-init-not-called
        self,
        time_zone: Optional[str] = None,
        time_unit: Optional[str] = None,
    ) -> None:
        object.__setattr__(self, "type", pl.Datetime(time_zone, time_unit))

    @classmethod
    def from_parametrized_dtype(cls, polars_dtype: pl.Datetime):
        """Convert a :class:`polars.Decimal` to
        a Pandera :class:`pandera.engines.polars_engine.Decimal`."""
        return cls(
            time_zone=polars_dtype.time_zone, time_unit=polars_dtype.time_unit
        )


@Engine.register_dtype(
    equivalents=[
        "time",
        datetime.time,
        pl.Time,
    ]
)
@immutable
class Time(DataType):
    """Polars time data type."""

    type = pl.Time


@Engine.register_dtype(
    equivalents=[
        "timedelta",
        datetime.timedelta,
        pl.Duration,
        dtypes.Timedelta,
        dtypes.Timedelta(),
    ]
)
@immutable(init=True)
class Timedelta(DataType, dtypes.Timedelta):
    """Polars timedelta data type."""

    type = pl.Duration

    def __init__(  # pylint:disable=super-init-not-called
        self,
        time_unit: Literal["ns", "us", "ms"] = "us",
    ) -> None:
        object.__setattr__(self, "type", pl.Duration(time_unit))

    @classmethod
    def from_parametrized_dtype(cls, polars_dtype: pl.Duration):
        """Convert a :class:`polars.Duration` to
        a Pandera :class:`pandera.engines.polars_engine.Duration`."""
        if polars_dtype.time_unit is None:
            return cls()
        return cls(time_unit=polars_dtype.time_unit)


###############################################################################
# Nested types
###############################################################################


###############################################################################
# Other types
###############################################################################


@Engine.register_dtype(
    equivalents=["bool", bool, pl.Boolean, dtypes.Bool, dtypes.Bool()]
)
@immutable
class Bool(DataType, dtypes.Bool):
    """Polars boolean data type."""

    type = pl.Boolean


@Engine.register_dtype(
    equivalents=["string", str, pl.Utf8, dtypes.String, dtypes.String()]
)
@immutable
class String(DataType, dtypes.String):
    """Polars string data type."""

    type = pl.Utf8


@Engine.register_dtype(equivalents=[pl.Categorical])
@immutable(init=True)
class Categorical(DataType):
    """Polars categorical data type."""

    type = pl.Categorical


@Engine.register_dtype(
    equivalents=["category", dtypes.Category, dtypes.Category()]
)
@immutable(init=True)
class Category(DataType, dtypes.Category):
    """Pandera categorical data type for polars."""

    type = pl.Utf8

    def __init__(  # pylint:disable=super-init-not-called
        self, categories: Optional[Iterable[Any]] = None
    ):
        dtypes.Category.__init__(self, categories, ordered=False)

    def coerce(self, data_container: PolarsObject) -> PolarsObject:
        """Coerce data container to the data type."""
        data_container = data_container.cast(self.type, strict=True)

        belongs_to_categories = self.__belongs_to_categories(data_container)

        if not check_polars_container_all_true(belongs_to_categories):
            raise ValueError(
                f"Could not coerce {type(data_container)} data_container "
                f"into type {self.type}. Invalid categories found in data_container."
            )
        return data_container

    def try_coerce(self, data_container: PolarsObject) -> PolarsObject:
        """Coerce data container to the data type,

        raises a :class:`~pandera.errors.ParserError` if the coercion fails
        :raises: :class:`~pandera.errors.ParserError`: if coercion fails
        """
        try:
            return self.coerce(data_container)
        except Exception as exc:  # pylint:disable=broad-except
            is_coercible: PolarsObject = polars_object_coercible(
                data_container, self.type
            ) & self.__belongs_to_categories(data_container)

            failure_cases = polars_failure_cases_from_coercible(
                data_container, is_coercible
            )
            raise errors.ParserError(
                f"Could not coerce {type(data_container)} data_container "
                f"into type {self.type}. Invalid categories found in data_container.",
                failure_cases=failure_cases,
            ) from exc

    def __belongs_to_categories(
        self, data_container: PolarsObject
    ) -> PolarsObject:
        if isinstance(data_container, pl.Series):
            belongs_to_categories = data_container.is_in(self.categories)
        else:
            belongs_to_categories = pl.DataFrame(
                {
                    column: data_container[column].is_in(self.categories)
                    for column in data_container.columns
                }
            )
        return belongs_to_categories

    def __str__(self):
        return "Category"


@Engine.register_dtype(equivalents=["null", pl.Null])
@immutable
class Null(DataType):
    """Polars null data type."""

    type = pl.Null


@Engine.register_dtype(equivalents=["object", object, pl.Object])
@immutable
class Object(DataType):
    """Semantic representation of a :class:`numpy.object_`."""

    type = pl.Object
