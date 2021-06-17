"""Pandas engine and data types."""
# pylint:disable=too-many-ancestors

# docstrings are inherited
# pylint:disable=missing-class-docstring

# pylint doesn't know about __init__ generated with dataclass
# pylint:disable=unexpected-keyword-arg,no-value-for-parameter
import builtins
import dataclasses
import datetime
import inspect
import platform
import warnings
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd

from .. import dtypes
from ..dtypes import immutable
from . import engine, numpy_engine

WINDOWS_PLATFORM = platform.system() == "Windows"

PandasObject = Union[pd.Series, pd.Index, pd.DataFrame]
PandasExtensionType = pd.core.dtypes.base.ExtensionDtype
PandasDtype = Union[pd.core.dtypes.base.ExtensionDtype, np.dtype, type]


def is_extension_dtype(pd_dtype: PandasDtype) -> bool:
    """Check if a value is a pandas extension type or instance of one."""
    return isinstance(pd_dtype, PandasExtensionType) or (
        isinstance(pd_dtype, type)
        and issubclass(pd_dtype, PandasExtensionType)
    )


@immutable(init=True)
class DataType(dtypes.DataType):
    """Base `DataType` for boxing Pandas data types."""

    type: Any = dataclasses.field(repr=False, init=False)

    def __init__(self, dtype: Any):
        super().__init__()
        object.__setattr__(self, "type", pd.api.types.pandas_dtype(dtype))
        dtype_cls = dtype if inspect.isclass(dtype) else dtype.__class__
        warnings.warn(
            f"'{dtype_cls}' support is not guaranteed.\n"
            + "Usage Tip: Consider writing a custom "
            + "pandera.dtypes.DataType or opening an issue at "
            + "https://github.com/pandera-dev/pandera"
        )

    def __post_init__(self):
        object.__setattr__(self, "type", pd.api.types.pandas_dtype(self.type))

    def coerce(self, data_container: PandasObject) -> PandasObject:
        return data_container.astype(self.type)

    def check(self, pandera_dtype: dtypes.DataType) -> bool:
        try:
            pandera_dtype = Engine.dtype(pandera_dtype)
        except TypeError:
            return False
        return super().check(pandera_dtype)

    def __str__(self) -> str:
        return str(self.type)

    def __repr__(self) -> str:
        return f"DataType({self})"


class Engine(  # pylint:disable=too-few-public-methods
    metaclass=engine.Engine,
    base_pandera_dtypes=(DataType, numpy_engine.DataType),
):
    """Pandas data type engine."""

    @classmethod
    def dtype(cls, data_type: Any) -> "DataType":
        """Convert input into a pandas-compatible
        Pandera :class:`DataType` object."""
        try:
            return engine.Engine.dtype(cls, data_type)
        except TypeError:
            if is_extension_dtype(data_type) and isinstance(data_type, type):
                try:
                    np_or_pd_dtype = data_type()
                    # Convert to str here because some pandas dtypes allow
                    # an empty constructor for compatibility but fail on
                    # str(). e.g: PeriodDtype
                    str(np_or_pd_dtype.name)
                except (TypeError, AttributeError) as err:
                    raise TypeError(
                        f" dtype {data_type} cannot be instantiated: {err}\n"
                        "Usage Tip: Use an instance or a string representation."
                    ) from None
            else:
                # let pandas transform any acceptable value
                # into a numpy or pandas dtype.
                np_or_pd_dtype = pd.api.types.pandas_dtype(data_type)
                if isinstance(np_or_pd_dtype, np.dtype):
                    np_or_pd_dtype = np_or_pd_dtype.type

            try:
                return engine.Engine.dtype(cls, np_or_pd_dtype)
            except TypeError as err:
                return DataType(np_or_pd_dtype)

    @classmethod
    def numpy_dtype(cls, pandera_dtype: dtypes.DataType) -> np.dtype:
        """Convert a pandera data type to a numpy data type."""
        pandera_dtype = engine.Engine.dtype(cls, pandera_dtype)

        alias = str(pandera_dtype).lower()
        if alias == "boolean":
            alias = "bool"
        elif alias == "string":
            alias = "str"
        return np.dtype(alias)


###############################################################################
# boolean
###############################################################################


Engine.register_dtype(
    numpy_engine.Bool,
    equivalents=["bool", bool, np.bool_, dtypes.Bool, dtypes.Bool()],
)


@Engine.register_dtype(
    equivalents=["boolean", pd.BooleanDtype, pd.BooleanDtype()],
)
@immutable
class Bool(DataType, dtypes.Bool):
    type = pd.BooleanDtype()


BOOL = Bool

###############################################################################
# number
###############################################################################


def _register_numpy_numbers(
    builtin_name: str, pandera_name: str, sizes: List[int]
) -> None:
    """Register pandera.engines.numpy_engine DataTypes
    with the pandas engine."""

    builtin_type = getattr(builtins, builtin_name, None)  # uint doesn't exist
    default_data = 1
    if builtin_type:
        default_data = builtin_type(default_data)
        default_pd_dtype = pd.Series([default_data]).dtype
    else:
        default_pd_dtype = pd.Series([default_data], dtype=builtin_name).dtype

    for bit_width in sizes:
        # e.g.: numpy.int64
        np_dtype = getattr(np, f"{builtin_name}{bit_width}")

        equivalents = set(
            (
                np_dtype,
                # e.g.: pandera.dtypes.Int64
                getattr(dtypes, f"{pandera_name}{bit_width}"),
                getattr(dtypes, f"{pandera_name}{bit_width}")(),
            )
        )

        add_default = True
        if WINDOWS_PLATFORM:
            print("ON WINDOWS PLATFORM")
            if builtin_name in {"int"} and bit_width == 64:
                print(f"ADDING {builtin_type}")
                equivalents.add(builtin_type)
                equivalents.add(builtin_name)
                if builtin_type is int:
                    equivalents.add("integer")
                equivalents |= set(
                    (
                        getattr(dtypes, pandera_name),
                        getattr(dtypes, pandera_name)(),
                    )
                )
            elif builtin_name in {"int"} and bit_width == 32:
                add_default = False

        if np_dtype == default_pd_dtype:
            equivalents |= set([default_pd_dtype])
            if add_default:
                equivalents |= set(
                    (
                        getattr(dtypes, pandera_name),
                        getattr(dtypes, pandera_name)(),
                    )
                )
            if builtin_type and add_default:
                equivalents.add(builtin_type)

            # results from pd.api.types.infer_dtype
            if builtin_type is float:
                equivalents.add("floating")
                equivalents.add("mixed-integer-float")
            elif builtin_type is int and add_default:
                equivalents.add("integer")

        numpy_data_type = getattr(numpy_engine, f"{pandera_name}{bit_width}")
        print(f"EQUIVALENTS FOR {numpy_data_type}: {list(equivalents)}")
        Engine.register_dtype(numpy_data_type, equivalents=list(equivalents))


###############################################################################
# signed integer
###############################################################################

_register_numpy_numbers(
    builtin_name="int",
    pandera_name="Int",
    sizes=[64, 32, 16, 8],
)


@Engine.register_dtype(equivalents=[pd.Int64Dtype, pd.Int64Dtype()])
@immutable
class Int64(DataType, dtypes.Int):
    type = pd.Int64Dtype()
    bit_width: int = 64


INT64 = Int64


@Engine.register_dtype(equivalents=[pd.Int32Dtype, pd.Int32Dtype()])
@immutable
class Int32(Int64):
    type = pd.Int32Dtype()
    bit_width: int = 32


INT32 = Int32


@Engine.register_dtype(equivalents=[pd.Int16Dtype, pd.Int16Dtype()])
@immutable
class Int16(Int32):
    type = pd.Int16Dtype()
    bit_width: int = 16


INT16 = Int16


@Engine.register_dtype(equivalents=[pd.Int8Dtype, pd.Int8Dtype()])
@immutable
class Int8(Int16):
    type = pd.Int8Dtype()
    bit_width: int = 8


INT8 = Int8

###############################################################################
# unsigned integer
###############################################################################

_register_numpy_numbers(
    builtin_name="uint",
    pandera_name="UInt",
    sizes=[64, 32, 16, 8],
)


@Engine.register_dtype(equivalents=[pd.UInt64Dtype, pd.UInt64Dtype()])
@immutable
class UInt64(DataType, dtypes.UInt):
    type = pd.UInt64Dtype()
    bit_width: int = 64


@Engine.register_dtype(equivalents=[pd.UInt32Dtype, pd.UInt32Dtype()])
@immutable
class UInt32(UInt64):
    type = pd.UInt32Dtype()
    bit_width: int = 32


@Engine.register_dtype(equivalents=[pd.UInt16Dtype, pd.UInt16Dtype()])
@immutable
class UInt16(UInt32):
    type = pd.UInt16Dtype()
    bit_width: int = 16


@Engine.register_dtype(equivalents=[pd.UInt8Dtype, pd.UInt8Dtype()])
@immutable
class UInt8(UInt16):
    type = pd.UInt8Dtype()
    bit_width: int = 8


UINT64 = UInt64
UINT32 = UInt32
UINT16 = UInt16
UINT8 = UInt8

# ###############################################################################
# # float
# ###############################################################################

_register_numpy_numbers(
    builtin_name="float",
    pandera_name="Float",
    sizes=[64, 32, 16] if WINDOWS_PLATFORM else [128, 64, 32, 16],
)

# ###############################################################################
# # complex
# ###############################################################################

_register_numpy_numbers(
    builtin_name="complex",
    pandera_name="Complex",
    sizes=[128, 64] if WINDOWS_PLATFORM else [256, 128, 64],
)

# ###############################################################################
# # nominal
# ###############################################################################


@Engine.register_dtype(
    equivalents=[
        "category",
        "categorical",
        dtypes.Category,
        pd.CategoricalDtype,
    ]
)
@immutable(init=True)
class Category(DataType, dtypes.Category):
    type: pd.CategoricalDtype = dataclasses.field(default=None, init=False)

    def __init__(  # pylint:disable=super-init-not-called
        self,
        categories: Optional[Iterable[Any]] = None,
        ordered: bool = False,
    ) -> None:
        dtypes.Category.__init__(self, categories, ordered)
        object.__setattr__(
            self,
            "type",
            pd.CategoricalDtype(self.categories, self.ordered),
        )

    @classmethod
    def from_parametrized_dtype(
        cls, cat: Union[dtypes.Category, pd.CategoricalDtype]
    ):
        """Convert a categorical to
        a Pandera :class:`~pandera.dtypes.pandas_engine.Category`."""
        return cls(  # type: ignore
            categories=cat.categories, ordered=cat.ordered
        )


@Engine.register_dtype(
    equivalents=["string", pd.StringDtype, pd.StringDtype()]
)
@immutable
class String(DataType, dtypes.String):
    type = pd.StringDtype()


STRING = String


@Engine.register_dtype(
    equivalents=["str", str, dtypes.String, dtypes.String(), np.str_]
)
@immutable
class NpString(numpy_engine.String):
    """Specializes numpy_engine.String.coerce to handle pd.NA values."""

    def coerce(self, data_container: PandasObject) -> np.ndarray:
        # Convert to object first to avoid
        # TypeError: object cannot be converted to an IntegerDtype
        data_container = data_container.astype(object)
        return data_container.where(
            data_container.isna(), data_container.astype(str)
        )

    def check(self, pandera_dtype: dtypes.DataType) -> bool:
        return isinstance(pandera_dtype, (numpy_engine.Object, type(self)))


Engine.register_dtype(
    numpy_engine.Object,
    equivalents=[
        "object",
        "O",
        "bytes",
        "decimal",
        "mixed-integer",
        "mixed",
        object,
        np.object_,
    ],
)

# ###############################################################################
# # time
# ###############################################################################


_PandasDatetime = Union[np.datetime64, pd.DatetimeTZDtype]


@Engine.register_dtype(
    equivalents=[
        "time",
        "datetime",
        "datetime64",
        datetime.datetime,
        np.datetime64,
        dtypes.Timestamp,
        dtypes.Timestamp(),
        pd.Timestamp,
    ]
)
@immutable(init=True)
class DateTime(DataType, dtypes.Timestamp):
    type: Optional[_PandasDatetime] = dataclasses.field(
        default=None, init=False
    )
    unit: str = "ns"
    tz: Optional[datetime.tzinfo] = None
    to_datetime_kwargs: Dict[str, Any] = dataclasses.field(
        default_factory=dict, compare=False, repr=False
    )

    def __post_init__(self):
        if self.tz is None:
            type_ = np.dtype("datetime64[ns]")
        else:
            type_ = pd.DatetimeTZDtype(self.unit, self.tz)
            # DatetimeTZDtype converted tz to tzinfo for us
            object.__setattr__(self, "tz", type_.tz)

        object.__setattr__(self, "type", type_)

    def coerce(self, data_container: PandasObject) -> PandasObject:
        def _to_datetime(col: pd.Series) -> pd.Series:
            col = pd.to_datetime(col, **self.to_datetime_kwargs)
            return col.astype(self.type)

        if isinstance(data_container, pd.DataFrame):
            # pd.to_datetime transforms a df input into a series.
            # We actually want to coerce every columns.
            return data_container.transform(_to_datetime)
        return _to_datetime(data_container)

    @classmethod
    def from_parametrized_dtype(cls, pd_dtype: pd.DatetimeTZDtype):
        """Convert a :class:`pandas.DatetimeTZDtype` to
        a Pandera :class:`~pandera.engines.pandas_engine.DateTime`."""
        return cls(unit=pd_dtype.unit, tz=pd_dtype.tz)  # type: ignore

    def __str__(self) -> str:
        if self.type == np.dtype("datetime64[ns]"):
            return "datetime64[ns]"
        return str(self.type)


Engine.register_dtype(
    numpy_engine.Timedelta64,
    equivalents=[
        "timedelta",
        "timedelta64",
        datetime.timedelta,
        np.timedelta64,
        pd.Timedelta,
        dtypes.Timedelta,
        dtypes.Timedelta(),
    ],
)


@Engine.register_dtype
@immutable(init=True)
class Period(DataType):
    """Representation of pandas :class:`pd.Period`."""

    type: pd.PeriodDtype = dataclasses.field(default=None, init=False)
    freq: Union[str, pd.tseries.offsets.DateOffset]

    def __post_init__(self):
        object.__setattr__(self, "type", pd.PeriodDtype(freq=self.freq))

    @classmethod
    def from_parametrized_dtype(cls, pd_dtype: pd.PeriodDtype):
        """Convert a :class:`pandas.PeriodDtype` to
        a Pandera :class:`~pandera.engines.pandas_engine.Period`."""
        return cls(freq=pd_dtype.freq)  # type: ignore


# ###############################################################################
# # misc
# ###############################################################################


@Engine.register_dtype(equivalents=[pd.SparseDtype])
@immutable(init=True)
class Sparse(DataType):
    """Representation of pandas :class:`pd.SparseDtype`."""

    type: pd.SparseDtype = dataclasses.field(default=None, init=False)
    dtype: PandasDtype = np.float_
    fill_value: Any = np.nan

    def __post_init__(self):
        object.__setattr__(
            self,
            "type",
            pd.SparseDtype(dtype=self.dtype, fill_value=self.fill_value),
        )

    @classmethod
    def from_parametrized_dtype(cls, pd_dtype: pd.SparseDtype):
        """Convert a :class:`pandas.SparseDtype` to
        a Pandera :class:`~pandera.engines.pandas_engine.Sparse`."""
        return cls(  # type: ignore
            dtype=pd_dtype.subtype, fill_value=pd_dtype.fill_value
        )


@Engine.register_dtype
@immutable(init=True)
class Interval(DataType):
    """Representation of pandas :class:`pd.IntervalDtype`."""

    type: pd.IntervalDtype = dataclasses.field(default=None, init=False)
    subtype: Union[str, np.dtype]

    def __post_init__(self):
        object.__setattr__(
            self, "type", pd.IntervalDtype(subtype=self.subtype)
        )

    @classmethod
    def from_parametrized_dtype(cls, pd_dtype: pd.IntervalDtype):
        """Convert a :class:`pandas.IntervalDtype` to
        a Pandera :class:`~pandera.engines.pandas_engine.Interval`."""
        return cls(subtype=pd_dtype.subtype)  # type: ignore


print("PANDAS ENGINE EQUIVALENTS")
for k, v in engine.Engine._registry[Engine].equivalents.items():
    print(f"{k}: equivalents={v}")
