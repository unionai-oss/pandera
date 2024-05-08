"""Polars engine and data types."""

import dataclasses
import datetime
import decimal
import inspect
import warnings
from typing import (
    Any,
    Iterable,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import polars as pl
from polars.datatypes import DataTypeClass, py_type_to_dtype
from polars.type_aliases import SchemaDict

from pandera import dtypes, errors
from pandera.api.polars.types import PolarsData
from pandera.constants import CHECK_OUTPUT_KEY
from pandera.dtypes import immutable
from pandera.engines import engine

PolarsDataContainer = Union[pl.LazyFrame, PolarsData]
PolarsDataType = Union[DataTypeClass, pl.DataType]

COERCION_ERRORS = (
    TypeError,
    pl.ArrowError,
    pl.InvalidOperationError,
    pl.ComputeError,
)


def polars_object_coercible(
    data_container: PolarsData, type_: PolarsDataType
) -> pl.LazyFrame:
    """Checks whether a polars object is coercible with respect to a type."""
    key = data_container.key or "*"
    coercible = data_container.lazyframe.cast(
        {key: type_}, strict=False
    ).select(pl.col(key).is_not_null())
    # reduce to a single boolean column
    return coercible.select(pl.all_horizontal(key).alias(CHECK_OUTPUT_KEY))


def polars_failure_cases_from_coercible(
    data_container: PolarsData,
    is_coercible: pl.LazyFrame,
) -> pl.LazyFrame:
    """Get the failure cases resulting from trying to coerce a polars object."""
    return data_container.lazyframe.with_context(is_coercible).filter(
        pl.col(CHECK_OUTPUT_KEY).not_()
    )


def polars_coerce_failure_cases(
    data_container: PolarsData,
    type_: Any,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Get the failure cases resulting from trying to coerce a polars object
    into particular data type.
    """
    try:
        is_coercible = polars_object_coercible(data_container, type_)
    except (TypeError, pl.InvalidOperationError):
        is_coercible = data_container.lazyframe.with_columns(
            **{CHECK_OUTPUT_KEY: pl.lit(False)}
        ).select(CHECK_OUTPUT_KEY)

    try:
        failure_cases = polars_failure_cases_from_coercible(
            data_container, is_coercible
        ).collect()
        is_coercible = is_coercible.collect()
    except COERCION_ERRORS:
        # If coercion fails, all of the relevant rows are failure cases
        failure_cases = data_container.lazyframe.select(
            data_container.key or "*"
        ).collect()

        is_coercible = (
            data_container.lazyframe.with_columns(
                **{CHECK_OUTPUT_KEY: pl.lit(False)}
            ).select(CHECK_OUTPUT_KEY)
        ).collect()

    return is_coercible, failure_cases


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

    def coerce(self, data_container: PolarsDataContainer) -> pl.LazyFrame:
        """Coerce data container to the data type."""
        if isinstance(data_container, pl.LazyFrame):
            data_container = PolarsData(data_container)

        if data_container.key is None:
            dtypes = self.type
        else:
            dtypes = {data_container.key: self.type}

        return data_container.lazyframe.cast(dtypes, strict=True)

    def try_coerce(self, data_container: PolarsDataContainer) -> pl.LazyFrame:
        """Coerce data container to the data type,
        raises a :class:`~pandera.errors.ParserError` if the coercion fails
        :raises: :class:`~pandera.errors.ParserError`: if coercion fails
        """
        if isinstance(data_container, pl.LazyFrame):
            data_container = PolarsData(data_container)

        try:
            lf = self.coerce(data_container)
            lf.collect()
            return lf
        except COERCION_ERRORS as exc:  # pylint:disable=broad-except
            _key = (
                ""
                if data_container.key is None
                else f"'{data_container.key}' in"
            )
            is_coercible, failure_cases = polars_coerce_failure_cases(
                data_container=data_container, type_=self.type
            )
            if data_container.key:
                failure_cases = failure_cases.select(data_container.key)
            raise errors.ParserError(
                f"Could not coerce {_key} LazyFrame with schema "
                f"{data_container.lazyframe.schema} "
                f"into type {self.type}",
                failure_cases=failure_cases,
                parser_output=is_coercible,
            ) from exc

    def check(
        self,
        pandera_dtype: dtypes.DataType,
        data_container: Optional[PolarsDataContainer] = None,
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

    type = pl.Decimal

    # polars Decimal doesn't have a rounding attribute
    rounding = None

    def __init__(  # pylint:disable=super-init-not-called
        self,
        precision: int = dtypes.DEFAULT_PYTHON_PREC,
        scale: int = 0,
    ) -> None:
        object.__setattr__(self, "precision", precision)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(
            self, "type", pl.Decimal(precision=precision, scale=scale)
        )

    @classmethod
    def from_parametrized_dtype(cls, polars_dtype: pl.Decimal):
        """Convert a :class:`polars.Decimal` to
        a Pandera :class:`pandera.engines.polars_engine.Decimal`."""
        return cls(precision=polars_dtype.precision, scale=polars_dtype.scale)

    def coerce(self, data_container: PolarsDataContainer) -> pl.LazyFrame:
        """Coerce data container to the data type."""
        if isinstance(data_container, pl.LazyFrame):
            data_container = PolarsData(data_container)

        key = data_container.key or "*"
        return data_container.lazyframe.cast({key: pl.Float64}).cast(
            {key: pl.Decimal(scale=self.scale, precision=self.precision)},
            strict=True,
        )

    def check(
        self,
        pandera_dtype: dtypes.DataType,
        data_container: Optional[PolarsDataContainer] = None,
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

    type: Type[pl.Datetime] = pl.Datetime
    time_zone_agnostic: bool = False

    def __init__(  # pylint:disable=super-init-not-called
        self,
        time_zone_agnostic: bool = False,
        time_zone: Optional[str] = None,
        time_unit: Optional[str] = None,
    ) -> None:

        _kwargs = {}
        if time_unit is not None:
            # avoid deprecated warning when initializing pl.Datetime:
            # passing time_unit=None is deprecated.
            _kwargs["time_unit"] = time_unit

        object.__setattr__(
            self, "type", pl.Datetime(time_zone=time_zone, **_kwargs)
        )
        object.__setattr__(self, "time_zone_agnostic", time_zone_agnostic)

    @classmethod
    def from_parametrized_dtype(cls, polars_dtype: pl.Datetime):
        """Convert a :class:`polars.Decimal` to
        a Pandera :class:`pandera.engines.polars_engine.Decimal`."""
        return cls(
            time_zone=polars_dtype.time_zone, time_unit=polars_dtype.time_unit
        )

    def check(
        self,
        pandera_dtype: dtypes.DataType,
        data_container: Optional[PolarsDataContainer] = None,
    ) -> Union[bool, Iterable[bool]]:
        try:
            pandera_dtype = Engine.dtype(pandera_dtype)
        except TypeError:
            return False

        if self.time_zone_agnostic:
            return (
                isinstance(pandera_dtype.type, pl.Datetime)
                and pandera_dtype.type.time_unit == self.type.time_unit
            )

        return self.type == pandera_dtype.type and super().check(pandera_dtype)


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


@Engine.register_dtype(equivalents=[pl.Array])
@immutable(init=True)
class Array(DataType):
    """Polars Array nested type."""

    type = pl.Array

    def __init__(  # pylint:disable=super-init-not-called
        self,
        inner: Optional[PolarsDataType] = None,
        width: Optional[int] = None,
    ) -> None:
        if inner or width:
            object.__setattr__(
                self, "type", pl.Array(inner=inner, width=width)
            )

    @classmethod
    def from_parametrized_dtype(cls, polars_dtype: pl.Array):
        return cls(inner=polars_dtype.inner, width=polars_dtype.width)


@Engine.register_dtype(equivalents=[pl.List])
@immutable(init=True)
class List(DataType):
    """Polars List nested type."""

    type = pl.List

    def __init__(  # pylint:disable=super-init-not-called
        self,
        inner: Optional[PolarsDataType] = None,
    ) -> None:
        if inner:
            object.__setattr__(self, "type", pl.List(inner=inner))

    @classmethod
    def from_parametrized_dtype(cls, polars_dtype: pl.List):
        return cls(inner=polars_dtype.inner)


@Engine.register_dtype(equivalents=[pl.Struct])
@immutable(init=True)
class Struct(DataType):
    """Polars Struct nested type."""

    type = pl.Struct

    def __init__(  # pylint:disable=super-init-not-called
        self,
        fields: Optional[Union[Sequence[pl.Field], SchemaDict]] = None,
    ) -> None:
        if fields:
            object.__setattr__(self, "type", pl.Struct(fields=fields))

    @classmethod
    def from_parametrized_dtype(cls, polars_dtype: pl.Struct):
        return cls(fields=polars_dtype.fields)


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

    def coerce(self, data_container: PolarsDataContainer) -> pl.LazyFrame:
        """Coerce data container to the data type."""
        if isinstance(data_container, pl.LazyFrame):
            data_container = PolarsData(data_container)

        lf = data_container.lazyframe.cast(self.type, strict=True)

        key = data_container.key or "*"
        belongs_to_categories = self.__belongs_to_categories(lf, key=key)

        all_true = (
            belongs_to_categories.select(pl.all_horizontal(key))
            .select(pl.all().all())
            .collect()
            .item()
        )
        if not all_true:
            raise ValueError(
                f"Could not coerce {type(lf)} data_container "
                f"into type {self.type}. Invalid categories found in data_container."
            )
        return lf

    def try_coerce(self, data_container: PolarsDataContainer) -> pl.LazyFrame:
        """Coerce data container to the data type,

        raises a :class:`~pandera.errors.ParserError` if the coercion fails
        :raises: :class:`~pandera.errors.ParserError`: if coercion fails
        """
        if isinstance(data_container, pl.LazyFrame):
            data_container = PolarsData(data_container)

        try:
            return self.coerce(data_container)
        except Exception as exc:  # pylint:disable=broad-except
            is_coercible: pl.LazyFrame = polars_object_coercible(
                data_container, self.type
            ) & self.__belongs_to_categories(
                data_container.lazyframe, key=data_container.key
            )

            failure_cases = polars_failure_cases_from_coercible(
                data_container, is_coercible
            )
            raise errors.ParserError(
                f"Could not coerce {type(data_container)} data_container "
                f"into type {self.type}. Invalid categories found in data_container.",
                failure_cases=failure_cases,
            ) from exc

    def __belongs_to_categories(
        self,
        lf: pl.LazyFrame,
        key: Optional[str] = None,
    ) -> pl.LazyFrame:
        return lf.select(pl.col(key or "*").is_in(self.categories))

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
