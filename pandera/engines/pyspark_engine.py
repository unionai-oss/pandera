"""PySpark engine and data types."""
# pylint:disable=too-many-ancestors

# docstrings are inherited
# pylint:disable=missing-class-docstring

# pylint doesn't know about __init__ generated with dataclass
# pylint:disable=unexpected-keyword-arg,no-value-for-parameter
import builtins
import dataclasses
import datetime
import decimal
import inspect
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Type,
    Union,
    cast,
)

from pydantic import BaseModel, ValidationError

from pandera import dtypes, errors
from pandera.dtypes import immutable
from pandera.engines import engine
import pyspark.sql.types as pst

try:
    import pyarrow  # pylint:disable=unused-import

    PYARROW_INSTALLED = True
except ImportError:
    PYARROW_INSTALLED = False

try:
    from typing import Literal  # type: ignore
except ImportError:
    from typing_extensions import Literal  # type: ignore


@immutable(init=True)
class DataType(dtypes.DataType):
    """Base `DataType` for boxing PySpark data types."""

    type: Any = dataclasses.field(repr=False, init=False)
    """Native pyspark dtype boxed by the data type."""

    def __init__(self, dtype: Any):
        super().__init__()
        object.__setattr__(self, "type", dtype)
        dtype_cls = dtype if inspect.isclass(dtype) else dtype.__class__
        warnings.warn(
            f"'{dtype_cls}' support is not guaranteed.\n"
            + "Usage Tip: Consider writing a custom "
            + "pandera.dtypes.DataType or opening an issue at "
            + "https://github.com/pandera-dev/pandera"
        )

    def __post_init__(self):
        # this method isn't called if __init__ is defined
        object.__setattr__(self, "type", self.type)  # pragma: no cover

    def check(
        self,
        pandera_dtype: dtypes.DataType,
    ) -> Union[bool, Iterable[bool]]:
        try:
            pandera_dtype = Engine.dtype(pandera_dtype)
        except TypeError:
            return False

        # attempts to compare pandas native type if possible
        # to let subclass inherit check
        # (super will compare that DataType classes are exactly the same)
        try:
            return self.type == pandera_dtype.type or super().check(pandera_dtype)
        except TypeError:
            return super().check(pandera_dtype)

    def __str__(self) -> str:
        return str(self.type)

    def __repr__(self) -> str:
        return f"DataType({self})"


class Engine(  # pylint:disable=too-few-public-methods
    metaclass=engine.Engine,
    base_pandera_dtypes=(DataType),
):
    """PySpark data type engine."""

    @classmethod
    def get_registered_dtypes():
        return list(
            pst.BooleanType,
            pst.StringType,
            pst.DateType,
            pst.IntegerType,
            pst.DoubleType,
        )

    @classmethod
    def dtype(cls, data_type: Any) -> dtypes.DataType:
        """Convert input into a pyspark-compatible
        Pandera :class:`~pandera.dtypes.DataType` object."""
        try:
            return engine.Engine.dtype(cls, data_type)
        except TypeError:
            pass
            # if isinstance(data_type, type):
            #     try:
            #         np_or_pd_dtype = data_type()
            #         # Convert to str here because some pandas dtypes allow
            #         # an empty constructor for compatibility but fail on
            #         # str(). e.g: PeriodDtype
            #         str(np_or_pd_dtype.name)
            #     except (TypeError, AttributeError) as err:
            #         raise TypeError(
            #             f" dtype {data_type} cannot be instantiated: {err}\n"
            #             "Usage Tip: Use an instance or a string "
            #             "representation."
            #         ) from None
            # else:
            #     # let pandas transform any acceptable value
            #     # into a numpy or pandas dtype.
            #     np_or_pd_dtype = pd.api.types.pandas_dtype(data_type)
            #     if isinstance(np_or_pd_dtype, np.dtype):
            #         # cast alias to platform-agnostic dtype
            #         # e.g.: np.intc -> np.int32
            #         common_np_dtype = np.dtype(np_or_pd_dtype.name)
            #         np_or_pd_dtype = common_np_dtype.type

            # return engine.Engine.dtype(cls, np_or_pd_dtype)


###############################################################################
# boolean
###############################################################################


Engine.register_dtype(
    pst.BooleanType,
    equivalents=["bool", bool, dtypes.Bool, dtypes.Bool()],
)


@immutable
class BOOL(DataType, dtypes.Bool):
    """Semantic representation of a :class:`pyspark.sql.types.BooleanType`."""

    type = pst.BooleanType()
    _bool_like = frozenset({True, False})

    def coerce_value(self, value: Any) -> Any:
        """Coerce an value to specified boolean type."""
        if value not in self._bool_like:
            raise TypeError(f"value {value} cannot be coerced to type {self.type}")
        return super().coerce_value(value)
