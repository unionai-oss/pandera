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

    # def check(
    #     self,
    #     pandera_dtype: dtypes.DataType,
    # ) -> Union[bool, Iterable[bool]]:
    #     try:
    #         pandera_dtype = Engine.dtype(pandera_dtype)
    #     except TypeError:
    #         return False

    #     # attempts to compare pandas native type if possible
    #     # to let subclass inherit check
    #     # (super will compare that DataType classes are exactly the same)
    #     try:
    #         return self.type == pandera_dtype.type or super().check(pandera_dtype)
    #     except TypeError:
    #         return super().check(pandera_dtype)

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
    def dtype(cls, data_type: Any) -> dtypes.DataType:
        """Convert input into a pyspark-compatible
        Pandera :class:`~pandera.dtypes.DataType` object."""
        try:
            return engine.Engine.dtype(cls, data_type)
        except TypeError:
            raise


###############################################################################
# boolean
###############################################################################


@Engine.register_dtype(
    equivalents=["bool", bool, dtypes.Bool, dtypes.Bool()],
)
@immutable
class Bool(DataType, dtypes.Bool):
    """Semantic representation of a :class:`pyspark.sql.types.BooleanType`."""

    type = pst.BooleanType()
    _bool_like = frozenset({True, False})

    def coerce_value(self, value: Any) -> Any:
        """Coerce an value to specified boolean type."""
        if value not in self._bool_like:
            raise TypeError(f"value {value} cannot be coerced to type {self.type}")
        return super().coerce_value(value)


@Engine.register_dtype(
    equivalents=["string", dtypes.String, dtypes.String()],  # type: ignore
)
@immutable
class String(DataType, dtypes.String):  # type: ignore
    """Semantic representation of a :class:`pyspark.sql.StringType`."""

    type = pst.StringType()  # type: ignore


@Engine.register_dtype(
    equivalents=["int", dtypes.Int, dtypes.Int()],  # type: ignore
)
@immutable
class Int(DataType, dtypes.Int):  # type: ignore
    """Semantic representation of a :class:`pyspark.sql.IntegerType`."""

    type = pst.IntegerType()  # type: ignore


@Engine.register_dtype(
    equivalents=["float", dtypes.String, dtypes.Float()],  # type: ignore
)
@immutable
class Float(DataType, dtypes.Float):  # type: ignore
    """Semantic representation of a :class:`pyspark.sql.FloatType`."""

    type = pst.FloatType()  # type: ignore
