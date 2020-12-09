"""Typing definitions and helpers."""
# pylint:disable=abstract-method,disable=too-many-ancestors
import sys
from typing import TYPE_CHECKING, Generic, Type, TypeVar

import pandas as pd
import typing_inspect

from .dtypes import PandasDtype, PandasExtensionType

if sys.version_info < (3, 8):  # pragma: no cover
    from typing_extensions import Literal
else:  # pragma: no cover
    from typing import Literal  # pylint:disable=no-name-in-module


_LEGACY_TYPING = sys.version_info[:3] < (3, 7, 0)

GenericDtype = TypeVar(  # type: ignore
    "GenericDtype",
    PandasDtype,
    PandasExtensionType,
    bool,
    int,
    str,
    float,
    Literal[PandasDtype.Bool],
    Literal[PandasDtype.DateTime],
    Literal[PandasDtype.Category],
    Literal[PandasDtype.Float],
    Literal[PandasDtype.Float16],
    Literal[PandasDtype.Float32],
    Literal[PandasDtype.Float64],
    Literal[PandasDtype.Int],
    Literal[PandasDtype.Int8],
    Literal[PandasDtype.Int16],
    Literal[PandasDtype.Int32],
    Literal[PandasDtype.Int64],
    Literal[PandasDtype.UInt8],
    Literal[PandasDtype.UInt16],
    Literal[PandasDtype.UInt32],
    Literal[PandasDtype.UInt64],
    Literal[PandasDtype.INT8],
    Literal[PandasDtype.INT16],
    Literal[PandasDtype.INT32],
    Literal[PandasDtype.INT64],
    Literal[PandasDtype.UINT8],
    Literal[PandasDtype.UINT16],
    Literal[PandasDtype.UINT32],
    Literal[PandasDtype.UINT64],
    Literal[PandasDtype.Object],
    Literal[PandasDtype.String],
    Literal[PandasDtype.STRING],
    Literal[PandasDtype.Timedelta],
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


if TYPE_CHECKING:  # pragma: no cover
    # pylint:disable=too-few-public-methods,invalid-name
    T = TypeVar("T")

    class DataFrame(pd.DataFrame, Generic[T]):
        """
        Representation of pandas.DataFrame, only used for type annotation.

        *new in 0.5.0*
        """


else:
    # pylint:disable=too-few-public-methods
    class DataFrame(pd.DataFrame, Generic[Schema]):
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
    """

    def __init__(self, raw_annotation: Type) -> None:
        self._parse_annotation(raw_annotation)

    @property
    def is_generic_df(self) -> bool:
        """True if the annotation is a pandera.typing.DataFrame."""
        return self.origin is not None and issubclass(self.origin, DataFrame)

    def _parse_annotation(self, raw_annotation: Type) -> None:
        """Parse key information from annotation.

        :param annotation: A subscripted type.
        :returns: Annotation
        """
        self.raw_annotation = raw_annotation

        self.optional = typing_inspect.is_optional_type(raw_annotation)
        if self.optional:
            # e.g: Typing.Union[pandera.typing.Index[str], NoneType]
            if _LEGACY_TYPING:  # pragma: no cover
                # get_args -> ((pandera.typing.Index, <class 'str'>), <class 'NoneType'>)
                self.origin, self.arg = typing_inspect.get_args(
                    raw_annotation
                )[0]
                return
            # get_args -> (pandera.typing.Index[str], <class 'NoneType'>)
            raw_annotation = typing_inspect.get_args(raw_annotation)[0]

        self.origin = typing_inspect.get_origin(raw_annotation)
        args = typing_inspect.get_args(raw_annotation)
        self.arg = args[0] if args else args

        self.literal = typing_inspect.is_literal_type(self.arg)
        if self.literal:
            self.arg = typing_inspect.get_args(self.arg)[0]


Bool = Literal[PandasDtype.Bool]  #: ``"bool"`` numpy dtype
DateTime = Literal[PandasDtype.DateTime]  #: ``"datetime64[ns]"`` numpy dtype
Timedelta = Literal[
    PandasDtype.Timedelta
]  #: ``"timedelta64[ns]"`` numpy dtype
Category = Literal[PandasDtype.Category]  #: pandas ``"categorical"`` datatype
Float = Literal[PandasDtype.Float]  #: ``"float"`` numpy dtype
Float16 = Literal[PandasDtype.Float16]  #: ``"float16"`` numpy dtype
Float32 = Literal[PandasDtype.Float32]  #: ``"float32"`` numpy dtype
Float64 = Literal[PandasDtype.Float64]  #: ``"float64"`` numpy dtype
Int = Literal[PandasDtype.Int]  #: ``"int"`` numpy dtype
Int8 = Literal[PandasDtype.Int8]  #: ``"int8"`` numpy dtype
Int16 = Literal[PandasDtype.Int16]  #: ``"int16"`` numpy dtype
Int32 = Literal[PandasDtype.Int32]  #: ``"int32"`` numpy dtype
Int64 = Literal[PandasDtype.Int64]  #: ``"int64"`` numpy dtype
UInt8 = Literal[PandasDtype.UInt8]  #: ``"uint8"`` numpy dtype
UInt16 = Literal[PandasDtype.UInt16]  #: ``"uint16"`` numpy dtype
UInt32 = Literal[PandasDtype.UInt32]  #: ``"uint32"`` numpy dtype
UInt64 = Literal[PandasDtype.UInt64]  #: ``"uint64"`` numpy dtype
INT8 = Literal[PandasDtype.INT8]  #: ``"Int8"`` pandas dtype:: pandas 0.24.0+
INT16 = Literal[PandasDtype.INT16]  #: ``"Int16"`` pandas dtype: pandas 0.24.0+
INT32 = Literal[PandasDtype.INT32]  #: ``"Int32"`` pandas dtype: pandas 0.24.0+
INT64 = Literal[PandasDtype.INT64]  #: ``"Int64"`` pandas dtype: pandas 0.24.0+
UINT8 = Literal[
    PandasDtype.UINT8
]  #: ``"UInt8"`` pandas dtype:: pandas 0.24.0+
UINT16 = Literal[
    PandasDtype.UINT16
]  #: ``"UInt16"`` pandas dtype: pandas 0.24.0+
UINT32 = Literal[
    PandasDtype.UINT32
]  #: ``"UInt32"`` pandas dtype: pandas 0.24.0+
UINT64 = Literal[
    PandasDtype.UINT64
]  #: ``"UInt64"`` pandas dtype: pandas 0.24.0+
Object = Literal[PandasDtype.Object]  #: ``"object"`` numpy dtype

String = Literal[PandasDtype.String]  #: ``"str"`` numpy dtype

#: ``"string"`` pandas dtypes: pandas 1.0.0+. For <1.0.0, this enum will
#: fall back on the str-as-object-array representation.
STRING = Literal[PandasDtype.STRING]  #: ``"str"`` numpy dtype
