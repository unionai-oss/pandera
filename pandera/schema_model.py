"""Class-based api"""
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
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
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
CheckOrHypothesis = Union[Check, Hypothesis]

_ValidatorConfig = namedtuple("_ValidatorConfig", ["fields", "regex", "check"])
_VALIDATOR_KEY = "__validator_config__"
_DATAFRAME_VALIDATOR_KEY = "__dataframe_validator_config__"
_TRANSFORMER_KEY = "__dataframe_transformer_config__"
_CONFIG_KEY = "Config"


def get_first_arg(annotation: Type) -> type:
    """Get first argument of subscripted type tp

    :example:

    >>> import numpy as np
    >>> from pandera.schema_model import Series, get_first_arg
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


class BaseConfig:
    name: Optional[str] = None
    coerce: bool = False
    strict: bool = False
    multiindex_coerce: bool = False
    multiindex_strict: bool = False
    multiindex_name: Optional[str] = None


_config_options = [attr for attr in vars(BaseConfig) if not attr.startswith("_")]


def _extract_config_options(config: Type) -> Dict[str, Any]:
    return {
        name: value for name, value in vars(config).items() if name in _config_options
    }


def _regex_filter(seq: Iterable, regexps: List[str]) -> Set[str]:
    """Filter items matching at least one of the regexes."""
    matched: Set[str] = set()
    for regex in regexps:
        pattern = re.compile(regex)
        matched.update(filter(pattern.match, seq))
    return matched


def _get_field_annotations(cls: Type["SchemaModel"]) -> Dict[str, Any]:
    annotations = cls.__annotations__
    if not annotations:
        raise SchemaInitError(f"{cls.__name__} is not annotated.")
    missing = []
    for attr_name, _ in inspect.getmembers(cls, lambda x: not inspect.isroutine(x)):
        if attr_name.startswith("_") or attr_name == _CONFIG_KEY:
            annotations.pop(attr_name, None)
        elif attr_name not in annotations:
            missing.append(attr_name)

    if missing:
        warnings.warn(
            f"The following unannotated attributes will be ignored: {missing}"
        )
    return annotations


def _update_checks(
    checks: Dict[str, List[CheckOrHypothesis]],
    fn: Callable,
    fields: List[str],
) -> Dict[str, List[CheckOrHypothesis]]:
    """Extract validator from function and append it to the checks list."""
    field_validator = getattr(fn, _VALIDATOR_KEY, None)
    if isinstance(field_validator, _ValidatorConfig):
        if field_validator.regex:
            matched = _regex_filter(fields, field_validator.fields)
        else:
            matched = field_validator.fields
        for field in matched:
            if field not in fields:
                raise SchemaInitError(
                    f"Validator {fn.__name__} is assigned to a non-existing "
                    + f"field '{field}'."
                )
            if field not in checks:
                checks[field] = []
            checks[field].append(field_validator.check)

    return checks


def _extract_df_check(fn: Callable) -> Optional[CheckOrHypothesis]:
    df_validator = getattr(fn, _DATAFRAME_VALIDATOR_KEY, None)
    if isinstance(df_validator, (Check, Hypothesis)):
        return df_validator
    return None


def _extract_transformer(fn: Callable) -> Optional[Callable]:
    if getattr(fn, _TRANSFORMER_KEY, None):
        return fn
    return None


def _build_schema_index(
    indexes: List[schema_components.Index], **multiindex_kwargs: Any
) -> Optional[SchemaIndex]:
    index: Optional[SchemaIndex] = None
    if indexes:
        if len(indexes) == 1:
            index = indexes[0]
            index._name = None  # don't force name on single index
        else:
            index = schema_components.MultiIndex(indexes, **multiindex_kwargs)
    return index


class SchemaModel:
    Config: Type[BaseConfig] = BaseConfig
    __schema__: Optional[DataFrameSchema] = None
    __config__: Type[BaseConfig] = BaseConfig

    def __new__(cls, *args, **kwargs):
        raise TypeError(f"{cls.__name__} may not be instantiated")

    @classmethod
    def to_schema(cls) -> DataFrameSchema:
        """Create DataFrameSchema from the SchemaModel."""
        if cls.__schema__:
            return cls.__schema__

        annotations = cls._inherit_field_annotations()
        checks: Dict[str, List[CheckOrHypothesis]] = {}
        df_checks: List[CheckOrHypothesis] = []
        transformer = None
        for _, fn in inspect.getmembers(cls, inspect.isfunction):
            _update_checks(checks, fn, list(annotations.keys()))
            df_check = _extract_df_check(fn)
            if df_check:
                df_checks.append(df_check)
            tr = _extract_transformer(fn)
            if transformer and tr:
                raise SchemaInitError(
                    f"{cls.__name__} can only have one 'dataframe_transformer'."
                )
            transformer = tr

        config_options = cls._inherit_config_options()
        mi_kwargs = {
            name[len("multiindex_") :]: value
            for name, value in config_options.items()
            if name.startswith("multiindex_")
        }
        columns, index = cls._build_columns_index(checks, annotations, **mi_kwargs)
        cls.__schema__ = DataFrameSchema(
            columns,
            index=index,
            checks=df_checks,
            transformer=transformer,
            coerce=config_options["coerce"],
            strict=config_options["strict"],
            name=config_options["name"],
        )
        return cls.__schema__

    @classmethod
    def _build_columns_index(
        cls,
        checks: Dict[str, List[CheckOrHypothesis]],
        annotations: Dict[str, Any],
        **multiindex_kwargs: Any,
    ) -> Tuple[
        Dict[str, schema_components.Column],
        Optional[Union[schema_components.Index, schema_components.MultiIndex]],
    ]:
        columns: Dict[str, schema_components.Column] = {}
        indexes: List[schema_components.Index] = []
        for field_name, annotation in annotations.items():
            optional = is_optional_type(annotation)
            if optional:
                # e.g extract Series[int] from Optional[Series[int]]
                annotation = get_first_arg(annotation)

            schema_component = get_origin(annotation)
            dtype = get_first_arg(annotation)
            field = getattr(cls, field_name, None)
            if field and not isinstance(field, FieldInfo):
                raise SchemaInitError(
                    f"'{field_name}' can only be assigned a 'Field', "
                    + f"not a '{field.__class__}.'"
                )
            field_checks = checks.get(field_name, [])

            if schema_component is Series:
                col_constructor = field.to_column if field else schema_components.Column
                columns[field_name] = col_constructor(
                    dtype,
                    required=not optional,
                    checks=field_checks,
                    name=field_name,
                )
            elif schema_component is Index:
                if optional:
                    raise SchemaInitError(f"Index '{field_name}' cannot be Optional.")
                index_constructor = field.to_index if field else schema_components.Index
                indexes.append(
                    index_constructor(dtype, checks=field_checks, name=field_name)
                )
            else:
                raise SchemaInitError(
                    f"Invalid annotation for {field_name}. "
                    + f"{annotation} should be of type Series or Index."
                )

        return columns, _build_schema_index(indexes, **multiindex_kwargs)

    @classmethod
    def _inherit_field_annotations(cls) -> Dict[str, Any]:
        """Collect field annotations from bases in mro reverse order."""
        bases = inspect.getmro(cls)[:-2]  # bases -> SchemaModel -> object
        bases = cast(Tuple[Type[SchemaModel]], bases)
        annotations = {}
        for base in reversed(bases):
            base_annotations = _get_field_annotations(base)
            annotations.update(base_annotations)
        return annotations

    @classmethod
    def _inherit_config_options(cls) -> Dict[str, Any]:
        """Collect config options from bases in mro reverse order."""
        bases = inspect.getmro(cls)[:-1]
        bases = cast(Tuple[Type[SchemaModel]], bases)
        root_model, *models = reversed(bases)

        options = _extract_config_options(root_model.Config)
        for model in models:
            config = getattr(model, _CONFIG_KEY, {})
            base_options = _extract_config_options(config)
            options.update(base_options)
        return options


Schema = TypeVar("Schema", bound=SchemaModel)


class DataFrame(pd.DataFrame, Generic[Schema]):
    """Representation of pandas.DataFrame."""


SchemaComponent = TypeVar("SchemaComponent", bound=SeriesSchemaBase)


def _to_checklist(checks: Optional[CheckList]) -> List[Union[Check, Hypothesis]]:
    checks = checks or []
    if isinstance(checks, (Check, Hypothesis)):
        return [checks]
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


AnyCallable = Callable[..., Any]
ClassValidator = Callable[[AnyCallable], Callable[..., bool]]


def validator(*fields, regex: bool = False, **check_kwargs) -> ClassValidator:
    """Decorate method on the SchemaModel indicating that it should be used to
    validate fields (columns or index).
    """

    def _wrapper(check_fn: Callable[..., bool]) -> Callable[..., bool]:
        check = Check(check_fn, **check_kwargs)
        setattr(
            check_fn,
            _VALIDATOR_KEY,
            _ValidatorConfig(set(fields), regex, check),
        )
        return check_fn

    return _wrapper


def dataframe_validator(_fn=None, **check_kwargs) -> ClassValidator:
    """Decorate method on the SchemaModel indicating that it should be used to
    validate the DataFrame.
    """

    def _wrapper(check_fn: Callable[..., bool]) -> Callable[..., bool]:
        check = Check(check_fn, **check_kwargs)
        setattr(
            check_fn,
            _DATAFRAME_VALIDATOR_KEY,
            check,
        )
        return check_fn

    if callable(_fn):
        return _wrapper(_fn)  # type: ignore
    return _wrapper


def dataframe_transformer(fn=None) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Decorate method on the SchemaModel indicating that it should be used to
    transform the DataFrame.
    """

    def _wrapper(
        check_fn: Callable[..., bool]
    ) -> Callable[[pd.DataFrame], pd.DataFrame]:
        setattr(
            check_fn,
            _TRANSFORMER_KEY,
            True,
        )
        return check_fn

    if callable(fn):
        return _wrapper(fn)
    return _wrapper
