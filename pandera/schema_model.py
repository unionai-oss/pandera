"""Define typing extensions."""
import inspect
import re
import warnings
from collections import namedtuple
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
)

import pandas as pd
from typing_inspect import get_args, get_forward_arg, get_origin, is_optional_type

from . import dtypes, schema_components
from .checks import Check
from .dtypes import PandasDtype
from .errors import SchemaInitError
from .hypotheses import Hypothesis
from .schemas import CheckList, DataFrameSchema, PandasDtypeInputTypes, SeriesSchemaBase

Dtype = TypeVar("Dtype", PandasDtype, dtypes.PandasExtensionType, bool, int, str, float)
SchemaIndex = Union[schema_components.Index, schema_components.MultiIndex]

_ValidatorConfig = namedtuple("_ValidatorConfig", ["fields", "regex", "check"])
_VALIDATOR_CONFIG_KEY = "__validator_config__"


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


def _filter_regexes(seq: Iterable, regexes: List[str]) -> Set[str]:
    matched: Set[str] = set()
    for regex in regexes:
        pattern = re.compile(regex)
        matched.update(filter(pattern.match, seq))
    return matched


class SchemaModel:
    __schema: Optional[DataFrameSchema] = None
    __checks: Dict[str, Set[Check]] = {}

    def __new__(cls, *args, **kwargs):
        raise TypeError(f"{cls.__name__} may not be instantiated")

    @classmethod
    def _get_attrs(cls):
        def not_callable(x):
            return (
                not inspect.ismethod(x)
                and not inspect.isfunction(x)
                and not inspect.isbuiltin(x)
            )

        private_attrs = {"_SchemaModel__checks", "_SchemaModel__schema"}
        return [
            attr
            for attr, _ in inspect.getmembers(cls, not_callable)
            if not attr.startswith("__") and attr not in private_attrs
        ]

    @classmethod
    def _check_missing_annotations(cls):
        annotations = cls.__annotations__
        if not annotations:
            raise SchemaInitError(f"{cls.__name__} is not annotated.")

        missing = [attr for attr in cls._get_attrs() if attr not in annotations.keys()]
        if missing:
            warnings.warn(
                f"The following unannotated attributes will be ignored: {missing}"
            )

    @classmethod
    def _extract_validators(cls):
        model_fields = list(cls.__annotations__.keys())
        for fn_name, fn in inspect.getmembers(cls, inspect.isfunction):
            check_definition = getattr(fn, _VALIDATOR_CONFIG_KEY, None)
            if isinstance(check_definition, _ValidatorConfig):
                if check_definition.regex:
                    matched = _filter_regexes(model_fields, check_definition.fields)
                else:
                    matched = check_definition.fields

                for field in matched:
                    if field not in model_fields:
                        raise SchemaInitError(
                            f"Validator {fn_name} is assigned to a non-existing "
                            f"field '{field}'."
                        )
                    if field in cls.__checks:
                        cls.__checks[field].add(check_definition.check)
                    else:
                        cls.__checks[field] = {check_definition.check}

    @classmethod
    def get_schema(cls) -> DataFrameSchema:
        """Create DataFrameSchema from the SchemaModel."""
        if cls.__schema:
            return cls.__schema

        cls._check_missing_annotations()
        cls._extract_validators()

        columns: Dict[str, schema_components.Column] = {}
        indexes: List[schema_components.Index] = []
        for field_name, annotation in cls.__annotations__.items():
            optional = is_optional_type(annotation)
            if optional:
                # e.g extract Series[int] from Optional[Series[int]]
                annotation = get_first_arg(annotation)

            schema_component = get_origin(annotation)
            dtype = get_first_arg(annotation)
            field = getattr(cls, field_name, None)
            if field and not isinstance(field, FieldInfo):
                raise SchemaInitError(
                    f"'{field_name}' can only be assigned the result of 'Field', "
                    f"not a '{field.__class__}.'"
                )
            checks = cls.__checks.get(field_name, None)

            if schema_component is Series:
                col_constructor = field.to_column if field else schema_components.Column
                columns[field_name] = col_constructor(
                    dtype, required=not optional, checks=checks, name=field_name
                )
            elif schema_component is Index:
                if optional:
                    raise SchemaInitError(f"Index '{field_name}' cannot be Optional.")
                index_constructor = field.to_index if field else schema_components.Index
                indexes.append(index_constructor(dtype, checks=checks, name=field_name))
            else:
                raise SchemaInitError(
                    f"Invalid annotation for {field_name}. "
                    f"{annotation} should be of type Series or Index."
                )

        index: Optional[SchemaIndex] = None
        if indexes:
            if len(indexes) == 1:
                index = indexes[0]
                index._name = None  # don't force name on single index
            else:
                index = schema_components.MultiIndex(indexes)

        cls.__schema = DataFrameSchema(columns, index=index)
        return cls.__schema


Schema = TypeVar("Schema", bound=SchemaModel)


class DataFrame(pd.DataFrame, Generic[Schema]):
    """Representation of pandas.DataFrame."""


SchemaComponent = TypeVar("SchemaComponent", bound=SeriesSchemaBase)


def _to_checklist(checks: Optional[CheckList]) -> List[Union[Check, Hypothesis]]:
    checks = checks or []
    if isinstance(checks, (Check, Hypothesis)):
        checks = [checks]
    return checks


class FieldInfo:
    """Captures extra information about a field."""

    __slots__ = ("checks", "nullable", "allow_duplicates", "coerce", "regex")

    def __init__(
        self,
        checks: Optional[CheckList] = None,
        nullable: bool = False,
        allow_duplicates: bool = True,
        coerce: bool = False,
        regex: bool = False,
    ) -> None:
        self.checks = _to_checklist(checks)
        self.nullable = nullable
        self.allow_duplicates = allow_duplicates
        self.coerce = coerce
        self.regex = regex

    def _to_schema_component(
        self,
        pandas_dtype: PandasDtypeInputTypes,
        component: Type[SchemaComponent],
        checks: CheckList = None,
        **kwargs: Any,
    ) -> SchemaComponent:
        checks = _to_checklist(checks)

        return component(pandas_dtype, checks=self.checks + checks, **kwargs)

    def to_column(
        self,
        pandas_dtype: PandasDtypeInputTypes,
        checks: CheckList = None,
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
            checks=checks,
        )

    def to_index(
        self,
        pandas_dtype: PandasDtypeInputTypes,
        checks: CheckList = None,
        name: str = None,
    ) -> schema_components.Index:
        """Create a schema_components.Index from a field."""
        return self._to_schema_component(
            pandas_dtype,
            schema_components.Index,
            nullable=self.nullable,
            allow_duplicates=self.allow_duplicates,
            coerce=self.coerce,
            name=name,
            checks=checks,
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


def validator(*fields, regex: bool = False, **check_kwargs) -> Callable:
    """Decorate method on the SchemaModel indicating that it should be used to 
    validate fields."""

    def decorator(check_fn: Callable) -> Callable:
        setattr(
            check_fn,
            _VALIDATOR_CONFIG_KEY,
            _ValidatorConfig(set(fields), regex, Check(check_fn, **check_kwargs)),
        )
        return check_fn

    return decorator
