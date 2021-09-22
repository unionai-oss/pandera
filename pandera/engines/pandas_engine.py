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
import warnings
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
from packaging import version

from .. import dtypes, errors
from ..dtypes import immutable
from ..system import FLOAT_128_AVAILABLE
from . import engine, numpy_engine, utils
from .type_aliases import PandasDataType, PandasExtensionType, PandasObject


def pandas_version():
    """Return the pandas version."""

    return version.parse(pd.__version__)


PANDAS_1_3_0_PLUS = pandas_version().release >= (1, 3, 0)

try:
    from typing import Literal  # type: ignore
except ImportError:
    from typing_extensions import Literal  # type: ignore


def is_extension_dtype(pd_dtype: PandasDataType) -> bool:
    """Check if a value is a pandas extension type or instance of one."""
    return isinstance(pd_dtype, PandasExtensionType) or (
        isinstance(pd_dtype, type)
        and issubclass(pd_dtype, PandasExtensionType)
    )


@immutable(init=True)
class DataType(dtypes.DataType):
    """Base `DataType` for boxing Pandas data types."""

    type: Any = dataclasses.field(repr=False, init=False)
    """Native pandas dtype boxed by the data type."""

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
        # this method isn't called if __init__ is defined
        object.__setattr__(
            self, "type", pd.api.types.pandas_dtype(self.type)
        )  # pragma: no cover

    def coerce(self, data_container: PandasObject) -> PandasObject:
        try:
            return data_container.astype(self.type)
        except (ValueError, TypeError) as exc:
            raise errors.ParserError(
                f"Could not coerce {type(data_container)} data_container "
                f"into type {self.type}",
                failure_cases=utils.numpy_pandas_coerce_failure_cases(
                    data_container, self.type
                ),
            ) from exc

    def check(self, pandera_dtype: dtypes.DataType) -> bool:
        try:
            pandera_dtype = Engine.dtype(pandera_dtype)
        except TypeError:
            return False

        # attempts to compare pandas native type if possible
        # to let subclass inherit check
        # (super will compare that DataType classes are exactly the same)
        try:
            return self.type == pandera_dtype.type or super().check(
                pandera_dtype
            )
        except TypeError:
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
        Pandera :class:`~pandera.dtypes.DataType` object."""
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
                        "Usage Tip: Use an instance or a string "
                        "representation."
                    ) from None
            else:
                # let pandas transform any acceptable value
                # into a numpy or pandas dtype.
                np_or_pd_dtype = pd.api.types.pandas_dtype(data_type)
                if isinstance(np_or_pd_dtype, np.dtype):
                    np_or_pd_dtype = np_or_pd_dtype.type

            return engine.Engine.dtype(cls, np_or_pd_dtype)

    @classmethod
    def numpy_dtype(cls, pandera_dtype: dtypes.DataType) -> np.dtype:
        """Convert a Pandera :class:`~pandera.dtypes.DataType
        to a :class:`numpy.dtype`."""
        pandera_dtype: dtypes.DataType = engine.Engine.dtype(
            cls, pandera_dtype
        )

        alias = str(pandera_dtype).lower()
        if alias == "boolean":
            alias = "bool"
        elif alias.startswith("string"):
            alias = "str"

        try:
            return np.dtype(alias)
        except TypeError as err:
            raise TypeError(
                f"Data type '{pandera_dtype}' cannot be cast to a numpy dtype."
            ) from err


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
class BOOL(DataType, dtypes.Bool):
    """Semantic representation of a :class:`pandas.BooleanDtype`."""

    type = pd.BooleanDtype()


###############################################################################
# number
###############################################################################


def _register_numpy_numbers(
    builtin_name: str, pandera_name: str, sizes: List[int]
) -> None:
    """Register pandera.engines.numpy_engine DataTypes
    with the pandas engine."""

    builtin_type = getattr(builtins, builtin_name, None)  # uint doesn't exist

    # default to int64 regardless of OS
    default_pd_dtype = {
        "int": np.dtype("int64"),
        "uint": np.dtype("uint64"),
    }.get(builtin_name, pd.Series([1], dtype=builtin_name).dtype)

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

        if np_dtype == default_pd_dtype:
            equivalents |= set(
                (
                    default_pd_dtype,
                    builtin_name,
                    getattr(dtypes, pandera_name),
                    getattr(dtypes, pandera_name)(),
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
class INT64(DataType, dtypes.Int):
    """Semantic representation of a :class:`pandas.Int64Dtype`."""

    type = pd.Int64Dtype()
    bit_width: int = 64


@Engine.register_dtype(equivalents=[pd.Int32Dtype, pd.Int32Dtype()])
@immutable
class INT32(INT64):
    """Semantic representation of a :class:`pandas.Int32Dtype`."""

    type = pd.Int32Dtype()
    bit_width: int = 32


@Engine.register_dtype(equivalents=[pd.Int16Dtype, pd.Int16Dtype()])
@immutable
class INT16(INT32):
    """Semantic representation of a :class:`pandas.Int16Dtype`."""

    type = pd.Int16Dtype()
    bit_width: int = 16


@Engine.register_dtype(equivalents=[pd.Int8Dtype, pd.Int8Dtype()])
@immutable
class INT8(INT16):
    """Semantic representation of a :class:`pandas.Int8Dtype`."""

    type = pd.Int8Dtype()
    bit_width: int = 8


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
class UINT64(DataType, dtypes.UInt):
    """Semantic representation of a :class:`pandas.UInt64Dtype`."""

    type = pd.UInt64Dtype()
    bit_width: int = 64


@Engine.register_dtype(equivalents=[pd.UInt32Dtype, pd.UInt32Dtype()])
@immutable
class UINT32(UINT64):
    """Semantic representation of a :class:`pandas.UInt32Dtype`."""

    type = pd.UInt32Dtype()
    bit_width: int = 32


@Engine.register_dtype(equivalents=[pd.UInt16Dtype, pd.UInt16Dtype()])
@immutable
class UINT16(UINT32):
    """Semantic representation of a :class:`pandas.UInt16Dtype`."""

    type = pd.UInt16Dtype()
    bit_width: int = 16


@Engine.register_dtype(equivalents=[pd.UInt8Dtype, pd.UInt8Dtype()])
@immutable
class UINT8(UINT16):
    """Semantic representation of a :class:`pandas.UInt8Dtype`."""

    type = pd.UInt8Dtype()
    bit_width: int = 8


# ###############################################################################
# # float
# ###############################################################################

_register_numpy_numbers(
    builtin_name="float",
    pandera_name="Float",
    sizes=[128, 64, 32, 16] if FLOAT_128_AVAILABLE else [64, 32, 16],
)

# ###############################################################################
# # complex
# ###############################################################################

_register_numpy_numbers(
    builtin_name="complex",
    pandera_name="Complex",
    sizes=[256, 128, 64] if FLOAT_128_AVAILABLE else [128, 64],
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
    """Semantic representation of a :class:`pandas.CategoricalDtype`."""

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
        a Pandera :class:`pandera.dtypes.pandas_engine.Category`."""
        return cls(  # type: ignore
            categories=cat.categories, ordered=cat.ordered
        )


if PANDAS_1_3_0_PLUS:

    @Engine.register_dtype(equivalents=["string", pd.StringDtype])
    @immutable(init=True)
    class STRING(DataType, dtypes.String):
        """Semantic representation of a :class:`pandas.StringDtype`."""

        type: pd.StringDtype = dataclasses.field(default=None, init=False)
        storage: Optional[Literal["python", "pyarrow"]] = "python"

        def __post_init__(self):
            type_ = pd.StringDtype(self.storage)
            object.__setattr__(self, "type", type_)

        @classmethod
        def from_parametrized_dtype(cls, pd_dtype: pd.StringDtype):
            """Convert a :class:`pandas.StringDtype` to
            a Pandera :class:`pandera.engines.pandas_engine.STRING`."""
            return cls(pd_dtype.storage)

        def __str__(self) -> str:
            return repr(self.type)


else:

    @Engine.register_dtype(
        equivalents=["string", pd.StringDtype, pd.StringDtype()]
    )  # type: ignore
    @immutable
    class STRING(DataType, dtypes.String):  # type: ignore
        """Semantic representation of a :class:`pandas.StringDtype`."""

        type = pd.StringDtype()


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
        a Pandera :class:`pandera.engines.pandas_engine.DateTime`."""
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
        a Pandera :class:`pandera.engines.pandas_engine.Period`."""
        return cls(freq=pd_dtype.freq)  # type: ignore


# ###############################################################################
# # misc
# ###############################################################################


@Engine.register_dtype(equivalents=[pd.SparseDtype])
@immutable(init=True)
class Sparse(DataType):
    """Representation of pandas :class:`pd.SparseDtype`."""

    type: pd.SparseDtype = dataclasses.field(default=None, init=False)
    dtype: PandasDataType = np.float_
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
        a Pandera :class:`pandera.engines.pandas_engine.Sparse`."""
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
        a Pandera :class:`pandera.engines.pandas_engine.Interval`."""
        return cls(subtype=pd_dtype.subtype)  # type: ignore


class PandasDtype(Enum):
    # pylint: disable=line-too-long,invalid-name
    """Enumerate all valid pandas data types.

    This class simply enumerates the valid numpy dtypes for pandas arrays.
    For convenience ``PandasDtype`` enums can all be accessed in the top-level
    ``pandera`` name space via the same enum name.

    .. warning::

        This class is deprecated and will be removed in pandera v0.9.0. Use
        python types, pandas type string aliases, numpy dtypes, or pandas
        dtypes instead. See :ref:`dtypes` for details.

    :examples:

    >>> import pandas as pd
    >>> import pandera as pa
    >>>
    >>>
    >>> pa.SeriesSchema(pa.PandasDtype.Int).validate(pd.Series([1, 2, 3]))
    0    1
    1    2
    2    3
    dtype: int64
    >>> pa.SeriesSchema(pa.PandasDtype.Float).validate(pd.Series([1.1, 2.3, 3.4]))
    0    1.1
    1    2.3
    2    3.4
    dtype: float64
    >>> pa.SeriesSchema(pa.PandasDtype.String).validate(pd.Series(["a", "b", "c"]))
    0    a
    1    b
    2    c
    dtype: object

    """

    # numpy data types
    Bool = "bool"  #: ``"bool"`` numpy dtype
    DateTime = "datetime64"  #: ``"datetime64[ns]"`` numpy dtype
    Timedelta = "timedelta64"  #: ``"timedelta64[ns]"`` numpy dtype
    Float = "float"  #: ``"float"`` numpy dtype
    Float16 = "float16"  #: ``"float16"`` numpy dtype
    Float32 = "float32"  #: ``"float32"`` numpy dtype
    Float64 = "float64"  #: ``"float64"`` numpy dtype
    Int = "int"  #: ``"int"`` numpy dtype
    Int8 = "int8"  #: ``"int8"`` numpy dtype
    Int16 = "int16"  #: ``"int16"`` numpy dtype
    Int32 = "int32"  #: ``"int32"`` numpy dtype
    Int64 = "int64"  #: ``"int64"`` numpy dtype
    UInt8 = "uint8"  #: ``"uint8"`` numpy dtype
    UInt16 = "uint16"  #: ``"uint16"`` numpy dtype
    UInt32 = "uint32"  #: ``"uint32"`` numpy dtype
    UInt64 = "uint64"  #: ``"uint64"`` numpy dtype
    Object = "object"  #: ``"object"`` numpy dtype
    Complex = "complex"  #: ``"complex"`` numpy dtype
    Complex64 = "complex64"  #: ``"complex"`` numpy dtype
    Complex128 = "complex128"  #: ``"complex"`` numpy dtype
    Complex256 = "complex256"  #: ``"complex"`` numpy dtype

    # pandas data types
    Category = "category"  #: pandas ``"categorical"`` datatype
    INT8 = "Int8"  #: ``"Int8"`` pandas dtype:: pandas 0.24.0+
    INT16 = "Int16"  #: ``"Int16"`` pandas dtype: pandas 0.24.0+
    INT32 = "Int32"  #: ``"Int32"`` pandas dtype: pandas 0.24.0+
    INT64 = "Int64"  #: ``"Int64"`` pandas dtype: pandas 0.24.0+
    UINT8 = "UInt8"  #: ``"UInt8"`` pandas dtype: pandas 0.24.0+
    UINT16 = "UInt16"  #: ``"UInt16"`` pandas dtype: pandas 0.24.0+
    UINT32 = "UInt32"  #: ``"UInt32"`` pandas dtype: pandas 0.24.0+
    UINT64 = "UInt64"  #: ``"UInt64"`` pandas dtype: pandas 0.24.0+
    String = "str"  #: ``"str"`` numpy dtype

    #: ``"string"`` pandas dtypes: pandas 1.0.0+. For <1.0.0, this enum will
    #: fall back on the str-as-object-array representation.
    STRING = "string"


# NOTE: This is a hack to raise a deprecation warning to show for users who
# are still using the PandasDtype enum.
# pylint:disable=invalid-name
class __PandasDtype__:
    def __init__(self):
        self.pandas_dtypes = PandasDtype

    def __getattr__(self, name):
        warnings.warn(
            "The PandasDtype class is deprecated and will be removed in "
            "pandera v0.9.0. Use python types, pandas type string aliases, "
            "numpy dtypes, or pandas dtypes instead.",
            DeprecationWarning,
        )
        return Engine.dtype(getattr(self.pandas_dtypes, name).value)

    def __iter__(self):
        for k in self.pandas_dtypes:
            yield k.name


_PandasDtype = __PandasDtype__()
