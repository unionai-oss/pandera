"""Common typing functionality."""
# pylint:disable=abstract-method,disable=too-many-ancestors

import inspect
from typing import TYPE_CHECKING, Any, Generic, Optional, Type, TypeVar

import pandas as pd
import typing_inspect

from .. import dtypes
from ..engines import numpy_engine, pandas_engine

Bool = dtypes.Bool  #: ``"bool"`` numpy dtype
DateTime = dtypes.DateTime  #: ``"datetime64[ns]"`` numpy dtype
Timedelta = dtypes.Timedelta  #: ``"timedelta64[ns]"`` numpy dtype
Category = dtypes.Category  #: pandas ``"categorical"`` datatype
Float = dtypes.Float  #: ``"float"`` numpy dtype
Float16 = dtypes.Float16  #: ``"float16"`` numpy dtype
Float32 = dtypes.Float32  #: ``"float32"`` numpy dtype
Float64 = dtypes.Float64  #: ``"float64"`` numpy dtype
Int = dtypes.Int  #: ``"int"`` numpy dtype
Int8 = dtypes.Int8  #: ``"int8"`` numpy dtype
Int16 = dtypes.Int16  #: ``"int16"`` numpy dtype
Int32 = dtypes.Int32  #: ``"int32"`` numpy dtype
Int64 = dtypes.Int64  #: ``"int64"`` numpy dtype
UInt8 = dtypes.UInt8  #: ``"uint8"`` numpy dtype
UInt16 = dtypes.UInt16  #: ``"uint16"`` numpy dtype
UInt32 = dtypes.UInt32  #: ``"uint32"`` numpy dtype
UInt64 = dtypes.UInt64  #: ``"uint64"`` numpy dtype
INT8 = pandas_engine.INT8  #: ``"Int8"`` pandas dtype:: pandas 0.24.0+
INT16 = pandas_engine.INT16  #: ``"Int16"`` pandas dtype: pandas 0.24.0+
INT32 = pandas_engine.INT32  #: ``"Int32"`` pandas dtype: pandas 0.24.0+
INT64 = pandas_engine.INT64  #: ``"Int64"`` pandas dtype: pandas 0.24.0+
UINT8 = pandas_engine.UINT8  #: ``"UInt8"`` pandas dtype:: pandas 0.24.0+
UINT16 = pandas_engine.UINT16  #: ``"UInt16"`` pandas dtype: pandas 0.24.0+
UINT32 = pandas_engine.UINT32  #: ``"UInt32"`` pandas dtype: pandas 0.24.0+
UINT64 = pandas_engine.UINT64  #: ``"UInt64"`` pandas dtype: pandas 0.24.0+
Object = numpy_engine.Object  #: ``"object"`` numpy dtype
String = dtypes.String  #: ``"str"`` numpy dtype
#: ``"string"`` pandas dtypes: pandas 1.0.0+. For <1.0.0, this enum will
#: fall back on the str-as-object-array representation.
STRING = pandas_engine.STRING  #: ``"str"`` numpy dtype
BOOL = pandas_engine.BOOL  #: ``"str"`` numpy dtype

try:
    Geometry = pandas_engine.Geometry  # : ``"geometry"`` geopandas dtype
    GEOPANDAS_INSTALLED = True
except AttributeError:
    GEOPANDAS_INSTALLED = False

if GEOPANDAS_INSTALLED:
    GenericDtype = TypeVar(  # type: ignore
        "GenericDtype",
        bool,
        int,
        str,
        float,
        pd.core.dtypes.base.ExtensionDtype,
        Bool,
        DateTime,
        Timedelta,
        Category,
        Float,
        Float16,
        Float32,
        Float64,
        Int,
        Int8,
        Int16,
        Int32,
        Int64,
        UInt8,
        UInt16,
        UInt32,
        UInt64,
        INT8,
        INT16,
        INT32,
        INT64,
        UINT8,
        UINT16,
        UINT32,
        UINT64,
        Object,
        String,
        STRING,
        Geometry,
        covariant=True,
    )
else:
    GenericDtype = TypeVar(  # type: ignore
        "GenericDtype",
        bool,
        int,
        str,
        float,
        pd.core.dtypes.base.ExtensionDtype,
        Bool,
        DateTime,
        Timedelta,
        Category,
        Float,
        Float16,
        Float32,
        Float64,
        Int,
        Int8,
        Int16,
        Int32,
        Int64,
        UInt8,
        UInt16,
        UInt32,
        UInt64,
        INT8,
        INT16,
        INT32,
        INT64,
        UINT8,
        UINT16,
        UINT32,
        UINT64,
        Object,
        String,
        STRING,
        covariant=True,
    )

Schema = TypeVar("Schema", bound="SchemaModel")  # type: ignore


# pylint:disable=invalid-name
if TYPE_CHECKING:
    T = TypeVar("T")  # pragma: no cover
else:
    T = Schema


class DataFrameBase(Generic[T]):
    # pylint: disable=too-few-public-methods
    """
    Pandera Dataframe base class for validating dataframes on
    initialization.
    """

    default_dtype: Optional[Type] = None

    def __setattr__(self, name: str, value: Any) -> None:
        # pylint: disable=no-member
        object.__setattr__(self, name, value)
        if name == "__orig_class__":
            orig_class = getattr(self, "__orig_class__")
            class_args = getattr(orig_class, "__args__", None)
            if class_args is not None and any(
                x.__name__ == "SchemaModel"
                for x in inspect.getmro(class_args[0])
            ):
                schema_model = value.__args__[0]

            # prevent the double validation problem by preventing checks for
            # dataframes with a defined pandera.schema
            pandera_accessor = getattr(self, "pandera")
            if (
                pandera_accessor.schema is None
                or pandera_accessor.schema != schema_model.to_schema()
            ):
                pandera_accessor.add_schema(schema_model.to_schema())
                self.__dict__ = schema_model.validate(self).__dict__


# pylint:disable=too-few-public-methods
class SeriesBase(Generic[GenericDtype]):
    """Pandera Series base class to use for all pandas-like APIs."""

    default_dtype: Optional[Type] = None

    def __get__(
        self, instance: object, owner: Type
    ) -> str:  # pragma: no cover
        raise AttributeError("Series should resolve to Field-s")


# pylint:disable=too-few-public-methods
class IndexBase(Generic[GenericDtype]):
    """Representation of pandas.Index, only used for type annotation.

    *new in 0.5.0*
    """

    default_dtype: Optional[Type] = None

    def __get__(
        self, instance: object, owner: Type
    ) -> str:  # pragma: no cover
        raise AttributeError("Indexes should resolve to pa.Index-s")


class AnnotationInfo:  # pylint:disable=too-few-public-methods
    """Captures extra information about an annotation.

    Attributes:
        origin: The non-parameterized generic class.
        arg: The first generic type (SchemaModel does not support more than
            1 argument).
        literal: Whether the annotation is a literal.
        optional: Whether the annotation is optional.
        raw_annotation: The raw annotation.
        metadata: Extra arguments passed to :data:`typing.Annotated`.
    """

    def __init__(self, raw_annotation: Type) -> None:
        self._parse_annotation(raw_annotation)

    @property
    def is_generic_df(self) -> bool:
        """True if the annotation is a DataFrameBase subclass."""
        try:
            if self.origin is None:
                return False
            return issubclass(self.origin, DataFrameBase)
        except TypeError:
            return False

    def _parse_annotation(self, raw_annotation: Type) -> None:
        """Parse key information from annotation.

        :param annotation: A subscripted type.
        :returns: Annotation
        """
        self.raw_annotation = raw_annotation
        self.origin = self.arg = None

        self.optional = typing_inspect.is_optional_type(raw_annotation)
        if self.optional and typing_inspect.is_union_type(raw_annotation):
            # Annotated with Optional or Union[..., NoneType]
            # get_args -> (pandera.typing.Index[str], <class 'NoneType'>)
            raw_annotation = typing_inspect.get_args(raw_annotation)[0]

        self.origin = typing_inspect.get_origin(raw_annotation)
        # Replace empty tuple returned from get_args by None
        args = typing_inspect.get_args(raw_annotation) or None
        self.arg = args[0] if args else args

        self.metadata = getattr(self.arg, "__metadata__", None)
        if self.metadata:
            self.arg = typing_inspect.get_args(self.arg)[0]

        self.literal = typing_inspect.is_literal_type(self.arg)
        if self.literal:
            self.arg = typing_inspect.get_args(self.arg)[0]

        self.default_dtype = getattr(raw_annotation, "default_dtype", None)
