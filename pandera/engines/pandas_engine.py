import builtins
import datetime
from dataclasses import field
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

from .. import dtypes_
from ..dtypes_ import DisableInitMixin, immutable
from . import engine, numpy_engine

PandasObject = Union[pd.Series, pd.Index, pd.DataFrame]
PandasExtensionType = pd.core.dtypes.base.ExtensionDtype


def is_extension_dtype(dtype):
    """Check if a value is a pandas extension type or instance of one."""
    return isinstance(dtype, PandasExtensionType) or (
        isinstance(dtype, type) and issubclass(dtype, PandasExtensionType)
    )


@immutable(init=True)
class DataType(dtypes_.DataType):
    type: Any = field(repr=False)

    def __post_init__(self):
        object.__setattr__(self, "type", pd.api.types.pandas_dtype(self.type))

    def coerce(self, obj: PandasObject) -> PandasObject:
        return obj.astype(self.type)

    def check(self, datatype: "DataType") -> bool:
        try:
            datatype = Engine.dtype(datatype)
        except TypeError:
            return False
        return super().check(datatype)

    def __str__(self) -> str:
        return str(self.type)

    def __repr__(self) -> str:
        return f"DataType({self})"


class Engine(
    metaclass=engine.Engine, base_datatype=(DataType, numpy_engine.DataType)
):
    @classmethod
    def dtype(cls, obj: Any) -> "DataType":
        try:
            return engine.Engine.dtype(cls, obj)
        except TypeError:
            if is_extension_dtype(obj) and isinstance(obj, type):
                try:
                    np_or_pd_dtype = obj()
                    # Convert to str here because some pandas dtypes allow
                    # an empty constructor for compatibility but fail on
                    # str(). e.g: PeriodDtype
                    str(np_or_pd_dtype.name)
                except (TypeError, AttributeError) as err:
                    raise TypeError(
                        f" dtype {obj} cannot be instantiated: {err}\n"
                        "Usage Tip: Use an instance or a string representation."
                    ) from None
            else:
                # let pandas transform any acceptable value
                # into a numpy or pandas dtype.
                np_or_pd_dtype = pd.api.types.pandas_dtype(obj)
                if isinstance(np_or_pd_dtype, np.dtype):
                    np_or_pd_dtype = np_or_pd_dtype.type

            try:
                return engine.Engine.dtype(cls, np_or_pd_dtype)
            except TypeError:
                return DataType(np_or_pd_dtype)


################################################################################
# boolean
################################################################################


Engine.register_dtype(
    numpy_engine.Bool,
    equivalents=["bool", bool, np.bool_, dtypes_.Bool, dtypes_.Bool()],
)


@Engine.register_dtype(
    equivalents=["boolean", pd.BooleanDtype, pd.BooleanDtype()],
)
@immutable
class Bool(DisableInitMixin, DataType, dtypes_.Bool):
    type = pd.BooleanDtype()


BOOL = Bool

################################################################################
# number
################################################################################


def _register_numpy_numbers(
    builtin_name: str, pandera_name: str, sizes: List[int]
) -> None:
    """Return a dict of equivalent builtin, numpy, pandera dtypes
    indexed by size in bits."""

    builtin_type = getattr(builtins, builtin_name, None)  # uint doesn't exist
    default_pd_dtype = pd.Series([1], dtype=builtin_name).dtype

    for bit_width in sizes:
        # e.g.: numpy.int64
        np_dtype = getattr(np, f"{builtin_name}{bit_width}")

        equivalents = set(
            (
                np_dtype,
                getattr(np, f"{builtin_name}{bit_width}"),
                # e.g.: pandera.dtypes.Int64
                getattr(dtypes_, f"{pandera_name}{bit_width}"),
                getattr(dtypes_, f"{pandera_name}{bit_width}")(),
            )
        )

        if np_dtype == default_pd_dtype:
            equivalents |= set(
                (
                    # e.g: numpy.int_
                    default_pd_dtype,
                    # e.g: pandera.dtypes.Int
                    getattr(dtypes_, pandera_name),
                    getattr(dtypes_, pandera_name)(),
                )
            )
            if builtin_type:
                equivalents.add(builtin_type)

            # results from pd.api.types.infer_dtype
            if builtin_type is float:
                equivalents.add("floating")
                equivalents.add("mixed-integer-float")
            elif builtin_type is int:
                equivalents.add("integer")

        numpy_data_type = getattr(numpy_engine, f"{pandera_name}{bit_width}")
        Engine.register_dtype(numpy_data_type, equivalents=equivalents)


################################################################################
## signed integer
################################################################################

_register_numpy_numbers(
    builtin_name="int",
    pandera_name="Int",
    sizes=[64, 32, 16, 8],
)


@Engine.register_dtype(equivalents=[pd.Int64Dtype, pd.Int64Dtype()])
@immutable
class Int64(DisableInitMixin, DataType, dtypes_.Int):
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

################################################################################
## unsigned integer
################################################################################

_register_numpy_numbers(
    builtin_name="uint",
    pandera_name="UInt",
    sizes=[64, 32, 16, 8],
)


@Engine.register_dtype(equivalents=[pd.UInt64Dtype, pd.UInt64Dtype()])
@immutable
class UInt64(DisableInitMixin, DataType, dtypes_.UInt):
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

# ################################################################################
# ## float
# ################################################################################

_register_numpy_numbers(
    builtin_name="float",
    pandera_name="Float",
    sizes=[128, 64, 32, 16],
)

# ################################################################################
# ## complex
# ################################################################################

_register_numpy_numbers(
    builtin_name="complex",
    pandera_name="Complex",
    sizes=[128, 64],
)

# ################################################################################
# # nominal
# ################################################################################


@Engine.register_dtype(
    equivalents=[
        "category",
        "categorical",
        dtypes_.Category,
        pd.CategoricalDtype,
    ]
)
@immutable(init=True)
class Category(DataType, dtypes_.Category):
    type: pd.CategoricalDtype = field(default=None, init=False)

    def __post_init__(self):
        dtypes_.Category.__post_init__(self)
        object.__setattr__(
            self,
            "type",
            pd.CategoricalDtype(self.categories, self.ordered),
        )

    @classmethod
    def from_parametrized_dtype(
        cls, cat: Union[dtypes_.Category, pd.CategoricalDtype]
    ):
        return cls(categories=cat.categories, ordered=cat.ordered)


@Engine.register_dtype(
    equivalents=["string", pd.StringDtype, pd.StringDtype()]
)
@immutable
class String(DisableInitMixin, DataType, dtypes_.String):
    type = pd.StringDtype()


STRING = String


@Engine.register_dtype(
    equivalents=["str", str, dtypes_.String, dtypes_.String(), np.str_]
)
@immutable
class NpString(numpy_engine.String):
    """Specializes numpy_engine.String.coerce to handle pd.NA values."""

    def coerce(self, obj: PandasObject) -> np.ndarray:
        # Convert to object first to avoid
        # TypeError: object cannot be converted to an IntegerDtype
        obj = obj.astype(object)
        return obj.where(obj.isna(), obj.astype(str))

    def check(self, datatype: "DataType") -> bool:
        return isinstance(datatype, (numpy_engine.Object, type(self)))


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

# ################################################################################
# # time
# ################################################################################
@Engine.register_dtype(
    equivalents=[
        "time",
        "datetime",
        "datetime64",
        datetime.date,
        datetime.datetime,
        np.datetime64,
        dtypes_.Timestamp,
        dtypes_.Timestamp(),
        pd.Timestamp,
    ]
)
@immutable(init=True)
class DateTime(DataType, dtypes_.Timestamp):
    type: Union[np.datetime64, pd.DatetimeTZDtype] = field(
        default=None, init=False
    )
    unit: str = "ns"
    tz: datetime.tzinfo = None
    to_datetime_kwargs: Dict[str, Any] = field(
        default=None, compare=False, repr=False
    )

    def __post_init__(self):
        if self.tz is None:
            type_ = np.dtype("datetime64")
        else:
            type_ = pd.DatetimeTZDtype(self.unit, self.tz)
            # DatetimeTZDtype converted tz to tzinfo for us
            object.__setattr__(self, "tz", type_.tz)

        object.__setattr__(self, "type", type_)

    def coerce(self, obj: PandasObject) -> PandasObject:
        kwargs = self.to_datetime_kwargs or {}

        def _to_datetime(col: pd.Series) -> pd.Series:
            return pd.to_datetime(col, **kwargs).astype(self.type)

        if isinstance(obj, pd.DataFrame):
            # pd.to_datetime transforms a df input into a series.
            # We actually want to coerce every columns.
            return obj.transform(_to_datetime)
        return _to_datetime(obj)

    @classmethod
    def from_parametrized_dtype(cls, pd_dtype: pd.DatetimeTZDtype):
        return cls(unit=pd_dtype.unit, tz=pd_dtype.tz)

    def __str__(self) -> str:
        if self.type == np.dtype("datetime64"):
            return "datetime64[ns]"
        return str(self.type)


@Engine.register_dtype(
    equivalents=[
        "time",
        "datetime",
        "datetime64",
        datetime.datetime,
        np.datetime64,
        dtypes_.Timestamp,
        dtypes_.Timestamp(),
        pd.Timestamp,
    ]
)
@immutable(init=True)
class DateTime(DataType, dtypes_.Timestamp):
    type: Union[np.datetime64, pd.DatetimeTZDtype] = field(
        default=None, init=False
    )
    unit: str = "ns"
    tz: datetime.tzinfo = None
    to_datetime_kwargs: Dict[str, Any] = field(
        default=None, compare=False, repr=False
    )

    def __post_init__(self):
        if self.tz is None:
            type_ = np.dtype("datetime64")
        else:
            type_ = pd.DatetimeTZDtype(self.unit, self.tz)
            # DatetimeTZDtype converted tz to tzinfo for us
            object.__setattr__(self, "tz", type_.tz)

        object.__setattr__(self, "type", type_)

    def coerce(self, obj: PandasObject) -> PandasObject:
        kwargs = self.to_datetime_kwargs or {}

        def _to_datetime(col: pd.Series) -> pd.Series:
            return pd.to_datetime(col, **kwargs).astype(self.type)

        if isinstance(obj, pd.DataFrame):
            # pd.to_datetime transforms a df input into a series.
            # We actually want to coerce every columns.
            return obj.transform(_to_datetime)
        return _to_datetime(obj)

    @classmethod
    def from_parametrized_dtype(cls, pd_dtype: pd.DatetimeTZDtype):
        return cls(unit=pd_dtype.unit, tz=pd_dtype.tz)

    def __str__(self) -> str:
        if self.type == np.dtype("datetime64"):
            return "datetime64[ns]"
        return str(self.type)


Engine.register_dtype(
    numpy_engine.DateTime64,
    equivalents=[
        "timedelta",
        "timedelta64",
        datetime.timedelta,
        np.timedelta64,
        pd.Timedelta,
        dtypes_.Timedelta,
        dtypes_.Timedelta(),
    ],
)


@Engine.register_dtype
@immutable(init=True)
class Period(DataType):
    type: pd.PeriodDtype = field(default=None, init=False)
    freq: Union[str, pd.tseries.offsets.DateOffset]

    def __post_init__(self):
        object.__setattr__(self, "type", pd.PeriodDtype(freq=self.freq))

    @classmethod
    def from_parametrized_dtype(cls, pd_dtype: pd.PeriodDtype):
        return cls(freq=pd_dtype.freq)


# ################################################################################
# # misc
# ################################################################################


@Engine.register_dtype(equivalents=[pd.SparseDtype])
@immutable(init=True)
class Sparse(DataType):
    type: pd.SparseDtype = field(default=None, init=False)
    dtype: Union[str, PandasExtensionType, np.dtype, "type"] = np.float_
    fill_value: Any = np.nan

    def __post_init__(self):
        object.__setattr__(
            self,
            "type",
            pd.SparseDtype(dtype=self.dtype, fill_value=self.fill_value),
        )

    @classmethod
    def from_parametrized_dtype(cls, pd_dtype: pd.SparseDtype):
        return cls(dtype=pd_dtype.subtype, fill_value=pd_dtype.fill_value)


@Engine.register_dtype
@immutable(init=True)
class Interval(DataType):
    type: pd.IntervalDtype = field(default=None, init=False)
    subtype: Union[str, np.dtype]

    def __post_init__(self):
        object.__setattr__(
            self, "type", pd.IntervalDtype(subtype=self.subtype)
        )

    @classmethod
    def from_parametrized_dtype(cls, pd_dtype: pd.IntervalDtype):
        return cls(subtype=pd_dtype.subtype)
