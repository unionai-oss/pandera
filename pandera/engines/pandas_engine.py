import builtins
import datetime
from dataclasses import field
from typing import Any, Dict, List, Union
import numpy as np
import pandas as pd

from pandera.engines.engine import Engine
import pandera.engines.numpy_engine
from pandera.engines.numpy_engine import *

import pandera.dtypes_
from ..dtypes_ import *
from .engine import Engine
import numpy as np

PandasObject = Union[pd.Series, pd.Index, pd.DataFrame]
PandasExtensionType = pd.core.dtypes.base.ExtensionDtype


def is_extension_dtype(dtype):
    """Check if a value is a pandas extension type or instance of one."""
    return isinstance(dtype, PandasExtensionType) or (
        isinstance(dtype, type) and issubclass(dtype, PandasExtensionType)
    )


@immutable(init=True)
class PandasDataType(DataType):
    type: Any = field(repr=False)

    def __post_init__(self):
        object.__setattr__(self, "type", pd.api.types.pandas_dtype(self.type))

    def coerce(self, obj: PandasObject) -> PandasObject:
        return obj.astype(self.type)

    def check(self, datatype: "PandasDataType") -> bool:
        try:
            datatype = PandasEngine.dtype(datatype)
        except TypeError:
            return False
        return super().check(datatype)

    def __str__(self) -> str:
        return str(self.type)

    def __repr__(self) -> str:
        return f"PandasDataType({self})"


class PandasEngine(
    metaclass=Engine, base_datatype=(PandasDataType, NumpyDataType)
):
    @classmethod
    def dtype(cls, obj: Any) -> "PandasDataType":
        try:
            return Engine.dtype(cls, obj)
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
                        f"Pandas dtype {obj} cannot be instantiated: {err}\n"
                        "Usage Tip: Use an instance or a string representation."
                    ) from None
            else:
                # let pandas transform any acceptable value
                # into a numpy or pandas dtype.
                np_or_pd_dtype = pd.api.types.pandas_dtype(obj)
                if isinstance(np_or_pd_dtype, np.dtype):
                    np_or_pd_dtype = np_or_pd_dtype.type

            try:
                return Engine.dtype(cls, np_or_pd_dtype)
            except TypeError:
                return PandasDataType(np_or_pd_dtype)


################################################################################
# boolean
################################################################################


PandasEngine.register_dtype(
    NumpyBool,
    equivalents=["bool", bool, np.bool_, Bool, Bool()],
)


@PandasEngine.register_dtype(
    equivalents=["boolean", pd.BooleanDtype, pd.BooleanDtype()],
)
@immutable
class PandasBool(DisableInitMixin, PandasDataType, Bool):
    type = pd.BooleanDtype()


BOOL = PandasBool

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
                getattr(pandera.dtypes_, f"{pandera_name}{bit_width}"),
                getattr(pandera.dtypes_, f"{pandera_name}{bit_width}")(),
            )
        )

        if np_dtype == default_pd_dtype:
            equivalents |= set(
                (
                    # e.g: numpy.int_
                    default_pd_dtype,
                    # e.g: pandera.dtypes.Int
                    getattr(pandera.dtypes_, pandera_name),
                    getattr(pandera.dtypes_, pandera_name)(),
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

        numpy_data_type = getattr(
            pandera.engines.numpy_engine, f"Numpy{pandera_name}{bit_width}"
        )
        PandasEngine.register_dtype(numpy_data_type, equivalents=equivalents)


################################################################################
## signed integer
################################################################################

_register_numpy_numbers(
    builtin_name="int",
    pandera_name="Int",
    sizes=[64, 32, 16, 8],
)


@PandasEngine.register_dtype(equivalents=[pd.Int64Dtype, pd.Int64Dtype()])
@immutable
class PandasInt64(DisableInitMixin, PandasDataType, Int):
    type = pd.Int64Dtype()
    bit_width: int = 64


INT64 = PandasInt64


@PandasEngine.register_dtype(equivalents=[pd.Int32Dtype, pd.Int32Dtype()])
@immutable
class PandasInt32(PandasInt64):
    type = pd.Int32Dtype()
    bit_width: int = 32


INT32 = PandasInt32


@PandasEngine.register_dtype(equivalents=[pd.Int16Dtype, pd.Int16Dtype()])
@immutable
class PandasInt16(PandasInt32):
    type = pd.Int16Dtype()
    bit_width: int = 16


INT16 = PandasInt16


@PandasEngine.register_dtype(equivalents=[pd.Int8Dtype, pd.Int8Dtype()])
@immutable
class PandasInt8(PandasInt16):
    type = pd.Int8Dtype()
    bit_width: int = 8


INT8 = PandasInt8

################################################################################
## unsigned integer
################################################################################

_register_numpy_numbers(
    builtin_name="uint",
    pandera_name="UInt",
    sizes=[64, 32, 16, 8],
)


@PandasEngine.register_dtype(equivalents=[pd.UInt64Dtype, pd.UInt64Dtype()])
@immutable
class PandasUInt64(DisableInitMixin, PandasDataType, UInt):
    type = pd.UInt64Dtype()
    bit_width: int = 64


@PandasEngine.register_dtype(equivalents=[pd.UInt32Dtype, pd.UInt32Dtype()])
@immutable
class PandasUInt32(PandasUInt64):
    type = pd.UInt32Dtype()
    bit_width: int = 32


@PandasEngine.register_dtype(equivalents=[pd.UInt16Dtype, pd.UInt16Dtype()])
@immutable
class PandasUInt16(PandasUInt32):
    type = pd.UInt16Dtype()
    bit_width: int = 16


@PandasEngine.register_dtype(equivalents=[pd.UInt8Dtype, pd.UInt8Dtype()])
@immutable
class PandasUInt8(PandasUInt16):
    type = pd.UInt8Dtype()
    bit_width: int = 8


UINT64 = PandasUInt64
UINT32 = PandasUInt32
UINT16 = PandasUInt16
UINT8 = PandasUInt8

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


@PandasEngine.register_dtype(
    equivalents=[
        "category",
        "categorical",
        Category,
        pd.CategoricalDtype,
    ]
)
@immutable(init=True)
class PandasCategorical(PandasDataType, Category):
    type: pd.CategoricalDtype = field(default=None, init=False)

    def __post_init__(self):
        Category.__post_init__(self)
        object.__setattr__(
            self,
            "type",
            pd.CategoricalDtype(self.categories, self.ordered),
        )

    @classmethod
    def from_parametrized_dtype(cls, cat: Union[Category, pd.CategoricalDtype]):
        return cls(categories=cat.categories, ordered=cat.ordered)


@PandasEngine.register_dtype(
    equivalents=["string", pd.StringDtype, pd.StringDtype()]
)
@immutable
class PandasString(DisableInitMixin, PandasDataType, String):
    type = pd.StringDtype()


STRING = PandasString


@PandasEngine.register_dtype(
    equivalents=["str", str, String, String(), np.str_]
)
@immutable
class PandasNpString(NumpyString):
    """Specializes NumpyString.coerce to handle pd.NA values."""

    def coerce(self, obj: PandasObject) -> np.ndarray:
        # Convert to object first to avoid
        # TypeError: object cannot be converted to an IntegerDtype
        obj = obj.astype(object)
        return obj.where(obj.isna(), obj.astype(str))

    def check(self, datatype: "DataType") -> bool:
        return isinstance(datatype, (NumpyObject, type(self)))


PandasEngine.register_dtype(
    NumpyObject,
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
@PandasEngine.register_dtype(
    equivalents=[
        "time",
        "datetime",
        "datetime64",
        datetime.date,
        datetime.datetime,
        np.datetime64,
        Timestamp,
        Timestamp(),
        pd.Timestamp,
    ]
)
@immutable(init=True)
class PandasDateTime(PandasDataType, Timestamp):
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


@PandasEngine.register_dtype(
    equivalents=[
        "time",
        "datetime",
        "datetime64",
        datetime.datetime,
        np.datetime64,
        Timestamp,
        Timestamp(),
        pd.Timestamp,
    ]
)
@immutable(init=True)
class PandasDateTime(PandasDataType, Timestamp):
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


PandasEngine.register_dtype(
    Timedelta,
    equivalents=[
        "timedelta",
        "timedelta64",
        datetime.timedelta,
        np.timedelta64,
        pd.Timedelta,
        Timedelta,
        Timedelta(),
    ],
)


@PandasEngine.register_dtype
@immutable(init=True)
class PandasPeriod(PandasDataType):
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


@PandasEngine.register_dtype(equivalents=[pd.SparseDtype])
@immutable(init=True)
class PandasSparse(PandasDataType):
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


@PandasEngine.register_dtype
@immutable(init=True)
class PandasInterval(PandasDataType):
    type: pd.IntervalDtype = field(default=None, init=False)
    subtype: Union[str, np.dtype]

    def __post_init__(self):
        object.__setattr__(self, "type", pd.IntervalDtype(subtype=self.subtype))

    @classmethod
    def from_parametrized_dtype(cls, pd_dtype: pd.IntervalDtype):
        return cls(subtype=pd_dtype.subtype)
