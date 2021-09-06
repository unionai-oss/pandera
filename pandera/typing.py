"""Typing definitions and helpers."""
# pylint:disable=abstract-method,disable=too-many-ancestors
from typing import TYPE_CHECKING, Generic, Type, TypeVar

import pandas as pd
import typing_inspect

from . import dtypes
from .engines import numpy_engine, pandas_engine

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


# pylint:disable=too-few-public-methods
class Index(pd.Index, Generic[GenericDtype]):
    """Representation of pandas.Index, only used for type annotation.

    *new in 0.5.0*
    """


# pylint:disable=too-few-public-methods
class Series(pd.Series, Generic[GenericDtype]):  # type: ignore
    """Representation of pandas.Series, only used for type annotation.

    *new in 0.5.0*
    """

    def __get__(
        self, instance: object, owner: Type
    ) -> str:  # pragma: no cover
        raise AttributeError("Series should resolve to Field-s")


# pylint:disable=invalid-name
if TYPE_CHECKING:
    T = TypeVar("T")  # pragma: no cover
else:
    T = Schema


# pylint:disable=too-few-public-methods
class DataFrame(pd.DataFrame, Generic[T]):
    """
    Representation of pandas.DataFrame, only used for type annotation.

    *new in 0.5.0*
    """


class AnnotationInfo:  # pylint:disable=too-few-public-methods
    """Captures extra information about an annotation.

    Attributes:
        origin: The non-parameterized generic class.
        arg: The first generic type (SchemaModel does not support more than 1 argument).
        literal: Whether the annotation is a literal.
        optional: Whether the annotation is optional.
        raw_annotation: The raw annotation.
        metadata: Extra arguments passed to :data:`typing.Annotated`.
    """

    def __init__(self, raw_annotation: Type) -> None:
        self._parse_annotation(raw_annotation)

    @property
    def is_generic_df(self) -> bool:
        """True if the annotation is a pandera.typing.DataFrame."""
        try:
            return self.origin is not None and issubclass(
                self.origin, DataFrame
            )
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
