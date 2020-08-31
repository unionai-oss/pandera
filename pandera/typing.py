"""Define typing extensions."""
import inspect
import warnings
from typing import Any, Dict, Generic, Iterable, List, Optional, Type, TypeVar

import pandas as pd
from typing_inspect import get_args, get_forward_arg, get_origin, is_optional_type

from . import dtypes, schema_components
from .checks import Check
from .dtypes import PandasDtype
from .errors import SchemaInitError
from .schemas import CheckList, DataFrameSchema, PandasDtypeInputTypes, SeriesSchemaBase

Dtype = TypeVar("Dtype", PandasDtype, dtypes.PandasExtensionType, bool, int, str, float)


def get_first_arg(annotation: Type) -> type:
    """Get first argument of subscripted type tp

    :example:

    >>> import numpy as np
    >>> from pandera.typing import Series, get_first_arg
    >>>
    >>> assert get_first_arg(Series[np.int32]) == np.int32
    >>> assert get_first_arg(Series["np.int32"]) == "np.int32"
    """
    arg = get_args(annotation)[0]
    # e.g get_args(Series["int32"])[0] gives ForwardRef('int32')
    fwd = get_forward_arg(arg)

    return fwd if fwd is not None else arg


class Series(pd.Series, Generic[Dtype]):
    """Representation of pandas.Series."""


class Index(pd.Index, Generic[Dtype]):
    """Representation of pandas.Index."""


class SchemaModel:
    _schema: Optional[DataFrameSchema] = None

    def __new__(cls, *args, **kwargs):
        raise TypeError(f"{cls.__name__} may not be instantiated")

    @classmethod
    def _check_missing_annotations(cls):
        annotations = cls.__annotations__
        if not annotations:
            raise SchemaInitError(f"{cls.__name__} is not annotated.")

        missing = []
        for name, value in inspect.getmembers(cls):
            if (
                not name.startswith("_")
                and not inspect.ismethod(value)
                and name not in annotations.keys()
            ):
                missing.append(name)

        if missing:
            warnings.warn(
                f"The following unannotated attributes will be ignored: {missing}"
            )

    @classmethod
    def get_schema(cls) -> DataFrameSchema:
        """Create DataFrameSchema from the SchemaModel."""
        if cls._schema:
            return cls._schema

        cls._check_missing_annotations()

        columns: Dict[str, schema_components.Index] = {}
        indexes: List[schema_components.Index] = []
        for arg_name, annotation in cls.__annotations__.items():
            optional = is_optional_type(annotation)
            if optional:
                # e.g extract Series[int] from Optional[Series[int]]
                annotation = get_first_arg(annotation)

            schema_component = get_origin(annotation)
            dtype = get_first_arg(annotation)
            field = getattr(cls, arg_name, None)
            if field and not isinstance(field, FieldInfo):
                raise SchemaInitError(
                    f"'{arg_name}' can only be assigned the result of 'Field', "
                    f"not a '{field.__class__}.'"
                )

            if schema_component is Series:
                col_constructor = field.to_column if field else schema_components.Column
                columns[arg_name] = col_constructor(
                    dtype, required=not optional, name=arg_name
                )
            elif schema_component is Index:
                if optional:
                    raise SchemaInitError(f"Index '{arg_name}' cannot be Optional.")
                index_constructor = field.to_index if field else schema_components.Index
                indexes.append(index_constructor(dtype, name=arg_name))
            else:
                raise SchemaInitError(
                    f"Invalid annotation for {arg_name}. "
                    f"{annotation} should be of type Series or Index."
                )

        if indexes:
            if len(indexes) == 1:
                index = indexes[0]
                index._name = None  # don't force name on single index
            else:
                index = schema_components.MultiIndex(indexes)
        else:
            index = None

        cls._schema = DataFrameSchema(columns, index=index)
        return cls._schema


Schema = TypeVar("Schema", bound=SchemaModel)


class DataFrame(pd.DataFrame, Generic[Schema]):
    """Representation of pandas.DataFrame."""


SchemaComponent = TypeVar("SchemaComponent", bound=SeriesSchemaBase)


class FieldInfo:
    """Captures extra information about a field."""

    __slots__ = ("checks", "nullable", "allow_duplicates", "coerce", "regex")

    def __init__(
        self,
        checks: CheckList = None,
        nullable: bool = False,
        allow_duplicates: bool = True,
        coerce: bool = False,
        regex: bool = False,
    ) -> None:
        self.checks = checks
        self.nullable = nullable
        self.allow_duplicates = allow_duplicates
        self.coerce = coerce
        self.regex = regex

    def _to_schema_component(
        self,
        pandas_dtype: PandasDtypeInputTypes,
        component: Type[SchemaComponent],
        **kwargs: Any,
    ) -> SchemaComponent:
        return component(pandas_dtype, checks=self.checks, **kwargs)

    def to_column(
        self,
        pandas_dtype: PandasDtypeInputTypes,
        required: bool = True,
        name: str = None,
    ) -> schema_components.Column:
        """Create a schema_components.Column from a field."""
        return self._to_schema_component(
            pandas_dtype,
            schema_components.Column,
            nullable=self.nullable,
            allow_duplicates=self.allow_duplicates,
            coerce=self.coerce,
            regex=self.regex,
            required=required,
            name=name,
        )

    def to_index(
        self, pandas_dtype: PandasDtypeInputTypes, name: str = None
    ) -> schema_components.Index:
        """Create a schema_components.Index from a field."""
        return self._to_schema_component(
            pandas_dtype,
            schema_components.Index,
            nullable=self.nullable,
            allow_duplicates=self.allow_duplicates,
            coerce=self.coerce,
            name=name,
        )


_check_dispatch = {
    "eq": Check.equal_to,
    "neq": Check.not_equal_to,
    "gt": Check.greater_than,
    "ge": Check.greater_than_or_equal_to,
    "lt": Check.less_than,
    "le": Check.less_than_or_equal_to,
    "in_range": Check.in_range,
    "isin": Check.isin,
    "notin": Check.notin,
    "str_contains": Check.str_contains,
    "str_endswith": Check.str_endswith,
    "str_matches": Check.str_matches,
    "str_length": Check.str_length,
    "str_startswith": Check.str_startswith,
}


def Field(
    *,
    eq: Any = None,
    neq: Any = None,
    gt: Any = None,
    ge: Any = None,
    lt: Any = None,
    le: Any = None,
    in_range: Dict[str, Any] = None,
    isin: Iterable = None,
    notin: Iterable = None,
    str_contains: str = None,
    str_endswith: str = None,
    str_length: Dict[str, Any] = None,
    str_matches: str = None,
    str_startswith: str = None,
    nullable: bool = False,
    allow_duplicates: bool = True,
    coerce: bool = False,
    regex: bool = False,
    ignore_na: bool = True,
    raise_warning: bool = False,
    n_failure_cases: int = 10,
) -> Any:
    """Used to provide extra information about a field of a SchemaModel.
    Some arguments apply only to number dtypes and some apply only to ``str``.
    """
    check_kwargs = {
        "ignore_na": ignore_na,
        "raise_warning": raise_warning,
        "n_failure_cases": n_failure_cases,
    }
    args = locals()
    checks = []
    for arg_name, check_constructor in _check_dispatch.items():
        arg_value = args[arg_name]
        if arg_value is None:
            continue
        if arg_name in {"in_range", "str_length"}:  # dict args
            check = check_constructor(**arg_value, **check_kwargs)
        else:
            check = check_constructor(arg_value, **check_kwargs)
        checks.append(check)

    return FieldInfo(
        checks=checks or None,
        nullable=nullable,
        allow_duplicates=allow_duplicates,
        coerce=coerce,
        regex=regex,
    )
