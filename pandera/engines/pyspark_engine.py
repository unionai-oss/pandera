"""PySpark engine and data types."""

# pylint:disable=too-many-ancestors,no-member

# docstrings are inherited
# pylint:disable=missing-class-docstring

# pylint doesn't know about __init__ generated with dataclass
# pylint:disable=unexpected-keyword-arg,no-value-for-parameter

import dataclasses
import inspect
import re
import sys
import warnings
from typing import Any, Iterable, Optional, Union

import pyspark
import pyspark.sql.types as pst
from packaging import version

from pandera import dtypes, errors
from pandera.dtypes import immutable
from pandera.engines import engine
from pandera.engines.type_aliases import PysparkObject

try:
    import pyarrow  # pylint:disable=unused-import

    PYARROW_INSTALLED = True
except ImportError:  # pragma: no cover
    PYARROW_INSTALLED = False


DEFAULT_PYSPARK_PREC = pst.DecimalType().precision
DEFAULT_PYSPARK_SCALE = pst.DecimalType().scale

CURRENT_PYSPARK_VERSION = version.parse(pyspark.__version__)


@immutable(init=True)
class DataType(dtypes.DataType):
    """Base `DataType` for boxing PySpark data types."""

    type: Any = dataclasses.field(repr=False, init=False)
    """Native pyspark dtype boxed by the data type."""

    def __init__(self, dtype: Any):
        super().__init__()
        # Pyspark str(<DataType>) doesnot return equivalent string using the below code to convert the datatype to class
        try:
            if isinstance(dtype, str):
                # To get the name of class from string with () at the end need to replace it
                regex = r"(\(\))"
                subst = ""
                # You can manually specify the number of replacements by changing the 4th argument
                dtype = re.sub(regex, subst, dtype, 0, re.MULTILINE)
                dtype = getattr(sys.modules["pyspark.sql.types"], dtype)()
        except AttributeError:  # pragma: no cover
            pass
        except TypeError:  # pragma: no cover
            pass

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
        data_container: Optional[Any] = None,  # pylint:disable=unused-argument
    ) -> Union[bool, Iterable[bool]]:
        try:
            return self.type == pandera_dtype.type
        except TypeError:  # pragma: no cover
            return False

    def __str__(self) -> str:
        return str(self.type)

    def __repr__(self) -> str:
        return f"DataType({self})"

    def coerce(self, data_container: PysparkObject) -> PysparkObject:
        """Pure coerce without catching exceptions."""
        coerced = data_container.astype(self.type)
        return coerced

    def try_coerce(self, data_container: PysparkObject) -> PysparkObject:
        try:
            coerced = self.coerce(data_container)
        except Exception as exc:  # pylint:disable=broad-except
            if isinstance(exc, errors.ParserError):
                raise
            type_alias = str(self)
            raise errors.ParserError(
                f"Could not coerce {type(data_container)} data_container "
                f"into type {type_alias}",
                failure_cases=None,
            ) from exc

        return coerced


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
            if isinstance(data_type, str):
                regex = r"(\(\d.*?\b\))"
                subst = ""
                # You can manually specify the number of replacements by changing the 4th argument
                data_type = re.sub(regex, subst, data_type, 0, re.MULTILINE)
            return engine.Engine.dtype(cls, data_type)
        except TypeError:  # pylint:disable=try-except-raise # pragma: no cover
            raise


###############################################################################
# boolean
###############################################################################


@Engine.register_dtype(
    equivalents=[
        bool,
        "bool",
        "BooleanType",
        "BooleanType()",
        pst.BooleanType(),
        pst.BooleanType,
    ],
)
@immutable
class Bool(DataType, dtypes.Bool):
    """Semantic representation of a :class:`pyspark.sql.types.BooleanType`."""

    type = pst.BooleanType()
    _bool_like = frozenset({True, False})

    def coerce_value(self, value: Any) -> Any:
        """Coerce an value to specified boolean type."""
        if value not in self._bool_like:  # pragma: no cover
            raise TypeError(
                f"value {value} cannot be coerced to type {self.type}"
            )
        return super().coerce_value(value)  # pragma: no cover


###############################################################################
# string
###############################################################################


@Engine.register_dtype(
    equivalents=[str, "str", "string", "StringType", "StringType()", pst.StringType(), pst.StringType],  # type: ignore
)
@immutable
class String(DataType, dtypes.String):  # type: ignore
    """Semantic representation of a :class:`pyspark.sql.types.StringType`."""

    type = pst.StringType()  # type: ignore


###############################################################################
# integer
###############################################################################


@Engine.register_dtype(
    equivalents=[int, "int", "IntegerType", "IntegerType()", pst.IntegerType(), pst.IntegerType],  # type: ignore
)
@immutable
class Int(DataType, dtypes.Int):  # type: ignore
    """Semantic representation of a :class:`pyspark.sql.types.IntegerType`."""

    type = pst.IntegerType()  # type: ignore


###############################################################################
# float
###############################################################################


@Engine.register_dtype(
    equivalents=[float, "float", "FloatType", "FloatType()", pst.FloatType(), pst.FloatType],  # type: ignore
)
@immutable
class Float(DataType, dtypes.Float):  # type: ignore
    """Semantic representation of a :class:`pyspark.sql.types.FloatType`."""

    type = pst.FloatType()  # type: ignore


###############################################################################
# bigint or long
###############################################################################


@Engine.register_dtype(
    equivalents=["bigint", "long", "LongType", "LongType()", pst.LongType(), pst.LongType],  # type: ignore
)
@immutable
class BigInt(DataType, dtypes.Int64):  # type: ignore
    """Semantic representation of a :class:`pyspark.sql.types.LongType`."""

    type = pst.LongType()  # type: ignore


###############################################################################
# smallint
###############################################################################


@Engine.register_dtype(
    equivalents=["smallint", "short", "ShortType", "ShortType()", pst.ShortType(), pst.ShortType],  # type: ignore
)
@immutable
class ShortInt(DataType, dtypes.Int16):  # type: ignore
    """Semantic representation of a :class:`pyspark.sql.types.ShortType`."""

    type = pst.ShortType()  # type: ignore


###############################################################################
# tinyint
###############################################################################


@Engine.register_dtype(
    equivalents=[bytes, "tinyint", "bytes", "ByteType", "ByteType()", pst.ByteType(), pst.ByteType],  # type: ignore
)
@immutable
class ByteInt(DataType, dtypes.Int8):  # type: ignore
    """Semantic representation of a :class:`pyspark.sql.types.ByteType`."""

    type = pst.ByteType()  # type: ignore


###############################################################################
# decimal
###############################################################################


@Engine.register_dtype(
    equivalents=["decimal", "DecimalType", "DecimalType()", pst.DecimalType(), pst.DecimalType],  # type: ignore
)
@immutable(init=True)
class Decimal(DataType, dtypes.Decimal):  # type: ignore
    """Semantic representation of a :class:`pyspark.sql.types.DecimalType`."""

    type: pst.DecimalType = dataclasses.field(default=pst.DecimalType, init=False)  # type: ignore[assignment]  # noqa

    def __init__(  # pylint:disable=super-init-not-called
        self,
        precision: int = DEFAULT_PYSPARK_PREC,
        scale: int = DEFAULT_PYSPARK_SCALE,
    ) -> None:
        dtypes.Decimal.__init__(self, precision, scale, None)
        object.__setattr__(
            self,
            "type",
            pst.DecimalType(self.precision, self.scale),  # type: ignore
        )

    @classmethod
    def from_parametrized_dtype(cls, ps_dtype: pst.DecimalType):
        """Convert a :class:`pyspark.sql.types.DecimalType` to
        a Pandera :class:`pandera.engines.pyspark_engine.Decimal`."""
        return cls(precision=ps_dtype.precision, scale=ps_dtype.scale)  # type: ignore

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

        # attempts to compare pyspark native type if possible
        # to let subclass inherit check
        # (super will compare that DataType classes are exactly the same)
        try:
            return (
                (self.type == pandera_dtype.type)
                & (self.scale == pandera_dtype.scale)
                & (self.precision == pandera_dtype.precision)
            )

        except TypeError:  # pragma: no cover
            return super().check(pandera_dtype)


###############################################################################
# double
###############################################################################


@Engine.register_dtype(
    equivalents=["double", "DoubleType", "DoubleType()", pst.DoubleType(), pst.DoubleType],  # type: ignore
)
@immutable
class Double(DataType, dtypes.Float):  # type: ignore
    """Semantic representation of a :class:`pyspark.sql.types.DoubleType`."""

    type = pst.DoubleType()


###############################################################################
# date
###############################################################################


@Engine.register_dtype(
    equivalents=["date", "DateType", "DateType()", pst.DateType(), pst.DateType],  # type: ignore
)
@immutable
class Date(DataType, dtypes.Date):  # type: ignore
    """Semantic representation of a :class:`pyspark.sql.types.DateType`."""

    type = pst.DateType()  # type: ignore


###############################################################################
# timestamp
###############################################################################

# Default timestamp equivalents
equivalents = ["datetime", "timestamp", "TimestampType", "TimestampType()", pst.TimestampType(), pst.TimestampType]  # type: ignore

# Include new Spark 3.4 TimestampNTZType as equivalents
if CURRENT_PYSPARK_VERSION >= version.parse("3.4"):
    equivalents += ["TimestampNTZType", "TimestampNTZType()", pst.TimestampNTZType, pst.TimestampNTZType()]  # type: ignore


@Engine.register_dtype(equivalents=equivalents)  # type: ignore
@immutable
class Timestamp(DataType, dtypes.Timestamp):  # type: ignore
    """Semantic representation of a :class:`pyspark.sql.types.TimestampType`."""

    type = pst.TimestampType()  # type: ignore


###############################################################################
# binary
###############################################################################


@Engine.register_dtype(
    equivalents=["binary", "BinaryType", "BinaryType()", pst.BinaryType(), pst.BinaryType],  # type: ignore
)
@immutable
class Binary(DataType, dtypes.Binary):  # type: ignore
    """Semantic representation of a :class:`pyspark.sql.types.BinaryType`."""

    type = pst.BinaryType()  # type: ignore


###############################################################################
# array
###############################################################################


@Engine.register_dtype(equivalents=[pst.ArrayType])
@immutable(init=True)
class ArrayType(DataType):
    """Semantic representation of a :class:`pyspark.sql.types.ArrayType`."""

    type: pst.ArrayType = dataclasses.field(default=pst.ArrayType, init=False)

    def __init__(  # pylint:disable=super-init-not-called
        self,
        elementType: Any = pst.StringType(),
        containsNull: bool = True,
    ) -> None:
        # super().__init__(self)
        object.__setattr__(self, "elementType", elementType)
        object.__setattr__(self, "containsNull", containsNull)

        object.__setattr__(
            self,
            "type",
            pst.ArrayType(self.elementType, self.containsNull),  # type: ignore
        )

    @classmethod
    def from_parametrized_dtype(cls, ps_dtype: pst.ArrayType):
        """Convert a :class:`pyspark.sql.types.ArrayType` to
        a Pandera :class:`pandera.engines.pyspark_engine.ArrayType`."""
        return cls(elementType=ps_dtype.elementType, containsNull=ps_dtype.containsNull)  # type: ignore

    def check(
        self,
        pandera_dtype: dtypes.DataType,
        data_container: Any = None,  # pylint:disable=unused-argument
    ) -> Union[bool, Iterable[bool]]:
        try:
            pandera_dtype = Engine.dtype(pandera_dtype)
        except TypeError:  # pragma: no cover
            return False
        # attempts to compare pyspark native type if possible
        # to let subclass inherit check
        # (super will compare that DataType classes are exactly the same)
        try:
            return (
                (self.type == pandera_dtype.type)
                & (self.type.elementType == pandera_dtype.type.elementType)
                & (self.type.containsNull == pandera_dtype.type.containsNull)
            )

        except TypeError:  # pragma: no cover
            return super().check(pandera_dtype)


###############################################################################
# map
###############################################################################


@Engine.register_dtype(equivalents=[pst.MapType])
@immutable(init=True)
class MapType(DataType):
    """Semantic representation of a :class:`pyspark.sql.types.MapType`."""

    type: pst.MapType = dataclasses.field(default=pst.MapType, init=False)

    def __init__(  # pylint:disable=super-init-not-called
        self,
        keyType: Any = pst.StringType(),
        valueType: Any = pst.StringType(),
        valueContainsNull: bool = True,
    ) -> None:
        # super().__init__(self)
        object.__setattr__(self, "keyType", keyType)
        object.__setattr__(self, "valueType", valueType)
        object.__setattr__(self, "valueContainsNull", valueContainsNull)

        object.__setattr__(
            self,
            "type",
            pst.MapType(self.keyType, self.valueType, self.valueContainsNull),  # type: ignore
        )

    @classmethod
    def from_parametrized_dtype(cls, ps_dtype: pst.MapType):
        """Convert a :class:`pyspark.sql.types.MapType` to
        a Pandera :class:`pandera.engines.pyspark_engine.MapType`."""
        return cls(
            keyType=ps_dtype.keyType,
            valueType=ps_dtype.valueType,
            valueContainsNull=ps_dtype.valueContainsNull,
        )  # type: ignore

    def check(
        self,
        pandera_dtype: dtypes.DataType,
        data_container: Any = None,  # pylint:disable=unused-argument
    ) -> Union[bool, Iterable[bool]]:
        try:
            pandera_dtype = Engine.dtype(pandera_dtype)
        except TypeError:  # pragma: no cover
            return False
        # attempts to compare pyspark native type if possible
        # to let subclass inherit check
        # (super will compare that DataType classes are exactly the same)
        try:
            return (
                (self.type == pandera_dtype.type)
                & (self.type.valueType == pandera_dtype.type.valueType)
                & (self.type.keyType == pandera_dtype.type.keyType)
                & (
                    self.type.valueContainsNull
                    == pandera_dtype.type.valueContainsNull
                )
            )

        except TypeError:  # pragma: no cover
            return super().check(pandera_dtype)
