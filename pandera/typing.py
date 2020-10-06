"""Typing definitions and helpers."""
# pylint:disable=abstract-method,disable=too-many-ancestors
import sys
from typing import Generic, Optional, Type, TypeVar, TYPE_CHECKING

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
    Literal[PandasDtype.Timedelta],
    covariant=True,
)
Schema = TypeVar("Schema", bound="SchemaModel")  # type: ignore


class Index(pd.Index, Generic[GenericDtype]):
    """Representation of pandas.Index."""


class Series(pd.Series, Generic[GenericDtype]):  # type: ignore # pylint:disable=too-many-ancestors
    """Representation of pandas.Series."""


if TYPE_CHECKING:
    DataFrame = pd.DataFrame
else:
    class DataFrame(pd.DataFrame, Generic[Schema]):
        """Representation of pandas.DataFrame."""


class AnnotationInfo:  # pylint:disable=too-few-public-methods
    """Captures extra information about an annotation."""

    def __init__(
        self, origin: Optional[Type], arg: Optional[Type], optional: bool, literal=False
    ) -> None:
        self.origin = origin
        self.arg = arg
        self.optional = optional
        self.literal = literal

    @property
    def is_generic_df(self) -> bool:
        """True if the annotation is a pandera.typing.DataFrame."""
        return self.origin is not None and issubclass(self.origin, DataFrame)


def parse_annotation(raw_annotation: Type) -> AnnotationInfo:
    """Parse key information from annotation.

    :param annotation: A subscripted type.
    :returns: Annotation
    """
    optional = typing_inspect.is_optional_type(raw_annotation)
    if optional:
        # e.g: Typing.Union[pandera.typing.Index[str], NoneType]
        if _LEGACY_TYPING:  # pragma: no cover
            # get_args -> ((pandera.typing.Index, <class 'str'>), <class 'NoneType'>)
            origin, arg = typing_inspect.get_args(raw_annotation)[0]
            return AnnotationInfo(origin, arg, optional)
        # get_args -> (pandera.typing.Index[str], <class 'NoneType'>)
        raw_annotation = typing_inspect.get_args(raw_annotation)[0]

    origin = typing_inspect.get_origin(raw_annotation)
    args = typing_inspect.get_args(raw_annotation)
    arg = args[0] if args else args

    literal = typing_inspect.is_literal_type(arg)
    if literal:
        arg = typing_inspect.get_args(arg)[0]

    return AnnotationInfo(origin=origin, arg=arg, optional=optional, literal=literal)


Bool = Literal[PandasDtype.Bool]
DateTime = Literal[PandasDtype.DateTime]
Category = Literal[PandasDtype.Category]
Float = Literal[PandasDtype.Float]
Float16 = Literal[PandasDtype.Float16]
Float32 = Literal[PandasDtype.Float32]
Float64 = Literal[PandasDtype.Float64]
Int = Literal[PandasDtype.Int]
Int8 = Literal[PandasDtype.Int8]
Int16 = Literal[PandasDtype.Int16]
Int32 = Literal[PandasDtype.Int32]
Int64 = Literal[PandasDtype.Int64]
UInt8 = Literal[PandasDtype.UInt8]
UInt16 = Literal[PandasDtype.UInt16]
UInt32 = Literal[PandasDtype.UInt32]
UInt64 = Literal[PandasDtype.UInt64]
INT8 = Literal[PandasDtype.INT8]
INT16 = Literal[PandasDtype.INT16]
INT32 = Literal[PandasDtype.INT32]
INT64 = Literal[PandasDtype.INT64]
UINT8 = Literal[PandasDtype.UINT8]
UINT16 = Literal[PandasDtype.UINT16]
UINT32 = Literal[PandasDtype.UINT32]
UINT64 = Literal[PandasDtype.UINT64]
Object = Literal[PandasDtype.Object]
String = Literal[PandasDtype.String]
Timedelta = Literal[PandasDtype.Timedelta]
