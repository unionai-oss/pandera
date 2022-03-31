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
from typing import Any, Dict, Iterable, List, Optional, Type, Union

import numpy as np
import pandas as pd
from packaging import version
from pydantic import BaseModel, ValidationError

from .. import dtypes, errors
from ..dtypes import immutable
from ..system import FLOAT_128_AVAILABLE
from . import engine, numpy_engine, utils
from .type_aliases import PandasDataType, PandasExtensionType, PandasObject


def pandas_version():
    """Return the pandas version."""

    return version.parse(pd.__version__)


PANDAS_1_2_0_PLUS = pandas_version().release >= (1, 2, 0)
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
        """Pure coerce without catching exceptions."""
        coerced = data_container.astype(self.type)
        if type(data_container).__module__.startswith("modin.pandas"):
            # NOTE: this is a hack to enable catching of errors in modin
            coerced.__str__()
        return coerced

    def coerce_value(self, value: Any) -> Any:
        """Coerce an value to a particular type."""
        # by default, the pandas Engine delegates to the underlying numpy
        # datatype to coerce a value to the correct type.
        return self.type.type(value)

    def try_coerce(self, data_container: PandasObject) -> PandasObject:
        try:
            return self.coerce(data_container)
        except Exception as exc:  # pylint:disable=broad-except
            if isinstance(exc, errors.ParserError):
                raise
            raise errors.ParserError(
                f"Could not coerce {type(data_container)} data_container "
                f"into type {self.type}",
                failure_cases=utils.numpy_pandas_coerce_failure_cases(
                    data_container, self
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
                    # cast alias to platform-agnostic dtype
                    # e.g.: np.intc -> np.int32
                    common_np_dtype = np.dtype(np_or_pd_dtype.name)
                    np_or_pd_dtype = common_np_dtype.type

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
    _bool_like = frozenset({True, False})

    def coerce_value(self, value: Any) -> Any:
        """Coerce an value to specified datatime type."""
        if value not in self._bool_like:
            raise TypeError(
                f"value {value} cannot be coerced to type {self.type}"
            )
        return super().coerce_value(value)


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

if PANDAS_1_2_0_PLUS:

    @Engine.register_dtype(equivalents=[pd.Float64Dtype, pd.Float64Dtype()])
    @immutable
    class FLOAT64(DataType, dtypes.Float):
        """Semantic representation of a :class:`pandas.Float64Dtype`."""

        type = pd.Float64Dtype()
        bit_width: int = 64

    @Engine.register_dtype(equivalents=[pd.Float32Dtype, pd.Float32Dtype()])
    @immutable
    class FLOAT32(FLOAT64):
        """Semantic representation of a :class:`pandas.Float32Dtype`."""

        type = pd.Float32Dtype()
        bit_width: int = 32


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
        self, categories: Optional[Iterable[Any]] = None, ordered: bool = False
    ) -> None:
        dtypes.Category.__init__(self, categories, ordered)
        object.__setattr__(
            self,
            "type",
            pd.CategoricalDtype(self.categories, self.ordered),
        )

    def coerce(self, data_container: PandasObject) -> PandasObject:
        """Pure coerce without catching exceptions."""
        coerced = data_container.astype(self.type)
        if (coerced.isna() & data_container.notna()).any(axis=None):
            raise TypeError(
                f"Data container cannot be coerced to type {self.type}"
            )
        if type(data_container).__module__.startswith("modin.pandas"):
            # NOTE: this is a hack to enable catching of errors in modin
            coerced.__str__()
        return coerced

    def coerce_value(self, value: Any) -> Any:
        """Coerce an value to a particular type."""
        if value not in self.categories:  # type: ignore
            raise TypeError(
                f"value {value} cannot be coerced to type {self.type}"
            )
        return value

    @classmethod
    def from_parametrized_dtype(
        cls, cat: Union[dtypes.Category, pd.CategoricalDtype]
    ):
        """Convert a categorical to
        a Pandera :class:`pandera.dtypes.pandas_engine.Category`."""
        return cls(categories=cat.categories, ordered=cat.ordered)  # type: ignore


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
        def _to_str(obj):
            # NOTE: this is a hack to handle the following case:
            # pyspark.pandas.Index doesn't support .where method yet, use numpy
            reverter = None
            if type(obj).__module__.startswith("pyspark.pandas"):
                # pylint: disable=import-outside-toplevel
                import pyspark.pandas as ps

                if isinstance(obj, ps.Index):
                    obj = obj.to_series()
                    reverter = ps.Index
            else:
                obj = obj.astype(object)

            obj = (
                obj.astype(str)
                if obj.notna().all(axis=None)
                else obj.where(obj.isna(), obj.astype(str))
            )
            return obj if reverter is None else reverter(obj)

        return _to_str(data_container)

    def check(self, pandera_dtype: dtypes.DataType) -> bool:
        return isinstance(pandera_dtype, (numpy_engine.Object, type(self)))


Engine.register_dtype(
    numpy_engine.Object,
    equivalents=[
        "object",
        "object_",
        "object0",
        "O",
        "bytes",
        "decimal",
        "mixed-integer",
        "mixed",
        "bytes",
        bytes,
        object,
        np.object_,
        np.bytes_,
        np.string_,
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
    """Semantic representation of a :class:`pandas.DatetimeTZDtype`."""

    type: Optional[_PandasDatetime] = dataclasses.field(
        default=None, init=False
    )
    unit: str = "ns"
    """The precision of the datetime data. Currently limited to "ns"."""
    tz: Optional[datetime.tzinfo] = None
    """The timezone."""
    to_datetime_kwargs: Dict[str, Any] = dataclasses.field(
        default_factory=dict, compare=False, repr=False
    )
    "Any additional kwargs passed to :func:`pandas.to_datetime` for coercion."

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
            # NOTE: this is a hack to support pyspark.pandas. This needs to be
            # thoroughly tested, right now pyspark.pandas returns NA when a
            # dtype value can't be coerced into the target dtype.
            to_datetime_fn = pd.to_datetime
            if type(col).__module__.startswith(
                "pyspark.pandas"
            ):  # pragma: no cover
                # pylint: disable=import-outside-toplevel
                import pyspark.pandas as ps

                to_datetime_fn = ps.to_datetime
            if type(col).__module__.startswith("modin.pandas"):
                # pylint: disable=import-outside-toplevel
                import modin.pandas as mpd

                to_datetime_fn = mpd.to_datetime

            col = to_datetime_fn(col, **self.to_datetime_kwargs)
            return col.astype(self.type)

        if isinstance(data_container, pd.DataFrame):
            # pd.to_datetime transforms a df input into a series.
            # We actually want to coerce every columns.
            return data_container.transform(_to_datetime)
        return _to_datetime(data_container)

    def coerce_value(self, value: Any) -> Any:
        """Coerce an value to specified datatime type."""
        if value is pd.NaT:
            return value
        return super().coerce_value(value)

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


# ###############################################################################
# # geopandas
# ###############################################################################

try:
    import geopandas as gpd

    GEOPANDAS_INSTALLED = True
except ImportError:  # pragma: no cover
    GEOPANDAS_INSTALLED = False

if GEOPANDAS_INSTALLED:

    @Engine.register_dtype(
        equivalents=[
            "geometry",
            gpd.array.GeometryDtype,
            gpd.array.GeometryDtype(),
        ]
    )
    @dtypes.immutable
    class Geometry(DataType):
        type = gpd.array.GeometryDtype()


###############################################################################
# pydantic
###############################################################################


@Engine.register_dtype
@dtypes.immutable(init=True)
class PydanticModel(DataType):
    """A pydantic model datatype applying to rows in a dataframe."""

    type: Type[BaseModel] = dataclasses.field(default=None, init=False)  # type: ignore # noqa

    # pylint:disable=super-init-not-called
    def __init__(self, model: Type[BaseModel]) -> None:
        object.__setattr__(self, "type", model)

    def coerce(self, data_container: pd.DataFrame) -> pd.DataFrame:
        """Coerce pandas dataframe with pydantic record model."""

        # pylint: disable=import-outside-toplevel
        from pandera import error_formatters

        def _coerce_row(row):
            """
            Coerce each row using pydantic model, keeping track of failure
            cases.
            """
            try:
                # pylint: disable=not-callable
                row = pd.Series(self.type(**row).dict())
                row["failure_cases"] = np.nan
            except ValidationError as exc:
                row["failure_cases"] = {
                    k: row[k] for k in (x["loc"][0] for x in exc.errors())
                }

            return row

        coerced_df = data_container.apply(_coerce_row, axis="columns")

        # raise a ParserError with failure cases where each case is a
        # dictionary containing the failed elements in the pydantic record
        if coerced_df["failure_cases"].any():
            failure_cases = coerced_df["failure_cases"][
                coerced_df["failure_cases"].notna()
            ].astype(str)
            raise errors.ParserError(
                f"Could not coerce {type(data_container)} data_container "
                f"into type {self.type}",
                failure_cases=error_formatters.reshape_failure_cases(
                    failure_cases, ignore_na=False
                ),
            )
        return coerced_df.drop(["failure_cases"], axis="columns")
