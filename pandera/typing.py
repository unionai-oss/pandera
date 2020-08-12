"""Define typing extensions."""
import inspect
import warnings
from typing import Generic, Literal, Type, TypeVar, Union, get_type_hints

import numpy as np
import pandas as pd
from typing_inspect import get_args, get_forward_arg, get_origin

from . import dtypes, schema_components
from .dtypes import PandasDtype
from .schemas import DataFrameSchema

Dtype = TypeVar("Dtype", str, PandasDtype, dtypes.PandasExtensionType)


def get_first_arg(tp) -> type:
    """Get first argument of subscripted type tp

    :example:

    >>> import numpy as np
    >>> from pandera.typing import Series, get_first_arg
    >>>
    >>> assert get_first_arg(Series[np.int32]) == np.int32
    >>> assert get_first_arg(Series["np.int32"]) == "np.int32"
    """
    arg = get_args(tp)[0]
    # e.g get_args(Series["int32"])[0] gives ForwardRef('int32')
    fwd = get_forward_arg(arg)

    return fwd if fwd is not None else arg


def is_frame_or_series_origin(annotation) -> bool:
    origin = get_origin(annotation)
    return origin is DataFrame or origin is Series


class Series(pd.Series, Generic[Dtype]):
    pass


class Index(pd.Index, Generic[Dtype]):
    pass


class SchemaModel:
    def __new__(cls, *args, **kwargs):
        raise TypeError(f"{cls.__name__} may not be instantiated")

    @classmethod
    def get_schema(cls) -> DataFrameSchema:
        columns = {}
        index = None
        # typing.get_type_hints evaluates the types
        # and chokes on ForwardReferences like Series['int32']
        hints = cls.__annotations__
        for arg_name, annotation in hints.items():
            origin = get_origin(annotation)
            dtype = get_first_arg(annotation)
            if origin is Series:
                columns[arg_name] = schema_components.Column(dtype)
            elif origin is Index:
                if index:
                    raise TypeError("Found multiple indexes.")
                index = schema_components.Index(dtype)
            else:
                raise TypeError(
                    f"Invalid annotation for {arg_name}. "
                    f"{annotation} should be of type Series or Index."
                )

        if not columns:
            raise TypeError(
                f"{cls.__name__} is empty. Did you annotate all class attributes?"
            )

        missing_annotations = []
        for name, value in inspect.getmembers(cls):
            if (
                not name.startswith("_")
                and not inspect.ismethod(value)
                and name not in hints.keys()
            ):
                missing_annotations.append(name)

        if missing_annotations:
            warnings.warn(
                "The following unnanotated attributes will be ignored: %s"
                % missing_annotations
            )

        return DataFrameSchema(columns, index=index)


Schema = TypeVar("Schema", bound=SchemaModel)


class DataFrame(pd.DataFrame, Generic[Schema]):
    pass
