"""SchemaModel components"""
from typing import (
    Any,
    Callable,
    Dict,
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

from .checks import Check
from .schema_components import (
    Column,
    Index,
    PandasDtypeInputTypes,
    SeriesSchemaBase,
)

AnyCallable = Callable[..., Any]
SchemaComponent = TypeVar("SchemaComponent", bound=SeriesSchemaBase)

CHECK_KEY = "__check_config__"
DATAFRAME_CHECK_KEY = "__dataframe_check_config__"

_CheckList = Union[Check, List[Check]]


def _to_checklist(checks: Optional[_CheckList]) -> List[Check]:
    checks = checks or []
    if isinstance(checks, Check):  # pragma: no cover
        return [checks]
    return checks


class FieldInfo:
    """Captures extra information about a field.

    *new in 0.5.0*
    """

    __slots__ = (
        "checks",
        "nullable",
        "allow_duplicates",
        "coerce",
        "regex",
        "check_name",
        "alias",
    )

    def __init__(
        self,
        checks: Optional[_CheckList] = None,
        nullable: bool = False,
        allow_duplicates: bool = True,
        coerce: bool = False,
        regex: bool = False,
        alias: str = None,
        check_name: bool = None,
    ) -> None:
        self.checks = _to_checklist(checks)
        self.nullable = nullable
        self.allow_duplicates = allow_duplicates
        self.coerce = coerce
        self.regex = regex
        self.alias = alias
        self.check_name = check_name

    def _to_schema_component(
        self,
        pandas_dtype: PandasDtypeInputTypes,
        component: Type[SchemaComponent],
        checks: _CheckList = None,
        **kwargs: Any,
    ) -> SchemaComponent:
        checks = self.checks + _to_checklist(checks)
        return component(pandas_dtype, checks=checks, **kwargs)  # type: ignore

    def to_column(
        self,
        pandas_dtype: PandasDtypeInputTypes,
        checks: _CheckList = None,
        required: bool = True,
        name: str = None,
    ) -> Column:
        """Create a schema_components.Column from a field."""
        return self._to_schema_component(
            pandas_dtype,
            Column,
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
        checks: _CheckList = None,
        name: str = None,
    ) -> Index:
        """Create a schema_components.Index from a field."""
        return self._to_schema_component(
            pandas_dtype,
            Index,
            nullable=self.nullable,
            allow_duplicates=self.allow_duplicates,
            coerce=self.coerce,
            name=name,
            checks=checks,
        )


def Field(
    *,
    eq: Any = None,
    ne: Any = None,
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
    alias: str = None,
    check_name: bool = None,
    **kwargs,
) -> Any:
    """Used to provide extra information about a field of a SchemaModel.

    *new in 0.5.0*

    Some arguments apply only to number dtypes and some apply only to ``str``.
    See the :ref:`User Guide <schema_models>` for more.

    :param alias: The public name of the column/index.

    :param check_name: Whether to check the name of the column/index during
        validation. `None` is the default behavior, which translates to `True`
        for columns and multi-index, and to `False` for a single index.
    :param kwargs: Specify custom checks that have been registered with the
        :class:`~pandera.extensions.register_check_method` decorator.
    """
    # pylint:disable=C0103,W0613,R0914
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
        if arg_name in {"in_range", "str_length"}:
            check_ = check_constructor(**arg_value, **check_kwargs)
        else:
            check_ = check_constructor(arg_value, **check_kwargs)
        checks.append(check_)

    return FieldInfo(
        checks=checks or None,
        nullable=nullable,
        allow_duplicates=allow_duplicates,
        coerce=coerce,
        regex=regex,
        check_name=check_name,
        alias=alias,
    )


_check_dispatch = {
    "eq": Check.equal_to,
    "ne": Check.not_equal_to,
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
    **Check.REGISTERED_CUSTOM_CHECKS,
}


class CheckInfo:  # pylint:disable=too-few-public-methods
    """Captures extra information about a Check."""

    def __init__(
        self,
        check_fn: AnyCallable,
        **check_kwargs: Any,
    ) -> None:
        self.check_fn = check_fn
        self.check_kwargs = check_kwargs

    def to_check(self, model_cls: Type) -> Check:
        """Create a Check from metadata."""
        name = self.check_kwargs.pop("name", None)
        if not name:
            name = getattr(
                self.check_fn, "__name__", self.check_fn.__class__.__name__
            )

        def _adapter(arg: Any) -> Union[bool, Iterable[bool]]:
            return self.check_fn(model_cls, arg)

        return Check(_adapter, name=name, **self.check_kwargs)


class FieldCheckInfo(CheckInfo):  # pylint:disable=too-few-public-methods
    """Captures extra information about a Check assigned to a field."""

    def __init__(
        self,
        fields: Set[str],
        check_fn: AnyCallable,
        regex: bool = False,
        **check_kwargs: Any,
    ) -> None:
        super().__init__(check_fn, **check_kwargs)
        self.fields = fields
        self.regex = regex


def _to_function_and_classmethod(
    fn: Union[AnyCallable, classmethod]
) -> Tuple[AnyCallable, classmethod]:
    if isinstance(fn, classmethod):
        fn, method = fn.__func__, cast(classmethod, fn)
    else:
        method = classmethod(fn)
    return fn, method


ClassCheck = Callable[[Union[classmethod, AnyCallable]], classmethod]


def check(*fields, regex: bool = False, **check_kwargs) -> ClassCheck:
    """Decorator to make SchemaModel method a column/index check function.

    *new in 0.5.0*

    This indicates that the decorated method should be used to validate a field
    (column or index). The method will be converted to a classmethod. Therefore
    its signature must start with `cls` followed by regular check arguments.
    See the :ref:`User Guide <schema_model_custom_check>` for more.

    :param _fn: Method to decorate.
    :param check_kwargs: Keywords arguments forwarded to Check.
    """

    def _wrapper(fn: Union[classmethod, AnyCallable]) -> classmethod:
        check_fn, check_method = _to_function_and_classmethod(fn)
        setattr(
            check_method,
            CHECK_KEY,
            FieldCheckInfo(set(fields), check_fn, regex, **check_kwargs),
        )
        return check_method

    return _wrapper


def dataframe_check(_fn=None, **check_kwargs) -> ClassCheck:
    """Decorator to make SchemaModel method a dataframe-wide check function.

    *new in 0.5.0*

    Decorate a method on the SchemaModel indicating that it should be used to
    validate the DataFrame. The method will be converted to a classmethod.
    Therefore its signature must start with `cls` followed by regular check
    arguments. See the :ref:`User Guide <schema_model_dataframe_check>` for
    more.

    :param check_kwargs: Keywords arguments forwarded to Check.
    """

    def _wrapper(fn: Union[classmethod, AnyCallable]) -> classmethod:
        check_fn, check_method = _to_function_and_classmethod(fn)
        setattr(
            check_method,
            DATAFRAME_CHECK_KEY,
            CheckInfo(check_fn, **check_kwargs),
        )
        return check_method

    if _fn:
        return _wrapper(_fn)  # type: ignore
    return _wrapper
