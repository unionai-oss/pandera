"""Typing definitions and helpers."""
# pylint: disable=R0903

import sys
from typing import Generic, Type, TypeVar, Union

import pandas as pd
import typing_inspect

from .dtypes import PandasDtype, PandasExtensionType

__all__ = ["Index", "Series", "DataFrame"]

_LEGACY_TYPING = sys.version_info[:3] < (3, 7, 0)

Dtype = Union[PandasDtype, PandasExtensionType, bool, int, str, float]
GenericDtype = TypeVar(
    "GenericDtype", PandasDtype, PandasExtensionType, bool, int, str, float
)
Schema = TypeVar("Schema", bound="SchemaModel")  # type: ignore


class Index(pd.Index, Generic[GenericDtype]):
    """Representation of pandas.Index."""


class Series(pd.Series, Generic[GenericDtype]):
    """Representation of pandas.Series."""


class DataFrame(pd.DataFrame, Generic[Schema]):
    """Representation of pandas.DataFrame."""


class Annotation:
    """Metadata extracted from an annotation."""
    def __init__(self, origin: Type, arg: Type, optional: bool) -> None:
        self.origin = origin
        self.arg = arg
        self.optional = optional

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({vars(self)})"


def is_frame_or_series_hint(raw_annotation: Type) -> bool:
    """Test if base annotation is a typing.Series or typing.DataFrame."""
    origin = typing_inspect.get_origin(raw_annotation)
    return origin is DataFrame or origin is Series


def parse_annotation(raw_annotation: Type) -> Annotation:
    """Parse key information from annotation.

    :param annotation: A subscripted type.
    :returns: Annotation
    """
    optional = typing_inspect.is_optional_type(raw_annotation)
    if optional:
        # e.g: Typing.Union[pandera.typing.Index[str], NoneType]
        if _LEGACY_TYPING:
            print(raw_annotation)
            # get_args -> ((pandera.typing.Index, <class 'str'>), <class 'NoneType'>)
            origin, arg = typing_inspect.get_args(raw_annotation)[0]
            return Annotation(origin, arg, optional)
        # get_args -> (pandera.typing.Index[str], <class 'NoneType'>)
        raw_annotation = typing_inspect.get_args(raw_annotation)[0]

    origin = typing_inspect.get_origin(raw_annotation)
    args = typing_inspect.get_args(raw_annotation)
    arg = args[0] if args else args
    return Annotation(origin=origin, arg=arg, optional=optional)
