"""DataFrameModel components"""

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

from pandera.api.base.model_components import (
    BaseCheckInfo,
    BaseFieldInfo,
    CheckArg,
    to_checklist,
)
from pandera.api.checks import Check
from pandera.api.pyspark.column_schema import ColumnSchema
from pandera.api.pyspark.components import Column
from pandera.api.pyspark.types import PySparkDtypeInputTypes
from pandera.errors import SchemaInitError

AnyCallable = Callable[..., Any]
SchemaComponent = TypeVar("SchemaComponent", bound=ColumnSchema)

CHECK_KEY = "__check_config__"
DATAFRAME_CHECK_KEY = "__dataframe_check_config__"


class FieldInfo(BaseFieldInfo):
    """Captures extra information about a field.

    *new in 0.5.0*
    """

    def _to_schema_component(
        self,
        dtype: PySparkDtypeInputTypes,
        component: Type[SchemaComponent],
        checks: CheckArg = None,
        **kwargs: Any,
    ) -> SchemaComponent:
        if self.dtype_kwargs:
            dtype = dtype(**self.dtype_kwargs)  # type: ignore
        checks = self.checks + to_checklist(checks)
        return component(dtype, checks=checks, **kwargs)  # type: ignore

    def to_column(
        self,
        dtype: PySparkDtypeInputTypes,
        checks: CheckArg = None,
        required: bool = True,
        name: str = None,
    ) -> Column:
        """Create a schema_components.Column from a field."""
        return self._to_schema_component(
            dtype,
            Column,
            nullable=self.nullable,
            coerce=self.coerce,
            regex=self.regex,
            required=required,
            name=name,
            checks=checks,
            title=self.title,
            description=self.description,
            metadata=self.metadata,
        )

    @property
    def properties(self) -> Dict[str, Any]:
        """Get column properties."""

        return {
            "dtype": self.dtype_kwargs,
            "checks": self.checks,
            "nullable": self.nullable,
            "coerce": self.coerce,
            "name": self.name,
            "regex": self.regex,
            "title": self.title,
            "description": self.description,
            "metadata": self.metadata,
        }


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
    str_contains: Optional[str] = None,
    str_endswith: Optional[str] = None,
    str_length: Optional[Dict[str, Any]] = None,
    str_matches: Optional[str] = None,
    str_startswith: Optional[str] = None,
    nullable: bool = False,
    unique: bool = False,
    coerce: bool = False,
    regex: bool = False,
    ignore_na: bool = True,
    raise_warning: bool = False,
    n_failure_cases: int = None,
    alias: Any = None,
    check_name: Optional[bool] = None,
    dtype_kwargs: Optional[Dict[str, Any]] = None,
    title: Optional[str] = None,
    description: Optional[str] = None,
    metadata: Optional[dict] = None,
    **kwargs,
) -> Any:
    """Used to provide extra information about a field of a DataFrameModel.

    *new in 0.16.0*

    Some arguments apply only to numeric dtypes and some apply only to ``str``.
    See the :ref:`User Guide <dataframe-models>` for more information.

    The keyword-only arguments from ``eq`` to ``str_startswith`` are dispatched
    to the built-in :py:class:`~pandera.api.checks.Check` methods.

    :param nullable: Whether or not the column/index can contain null values.
    :param unique: Whether column values should be unique. Currently Not supported
    :param coerce: coerces the data type if ``True``.
    :param regex: whether or not the field name or alias is a regex pattern.
    :param ignore_na: whether or not to ignore null values in the checks.
    :param raise_warning: raise a warning instead of an Exception.
    :param n_failure_cases: report the first n unique failure cases. If None,
        report all failure cases.
    :param alias: The public name of the column/index.
    :param check_name: Whether to check the name of the column/index during
        validation. `None` is the default behavior, which translates to `True`
        for columns and multi-index, and to `False` for a single index.
    :param dtype_kwargs: The parameters to be forwarded to the type of the
        field.
    :param title: A human-readable label for the field.
    :param description: An arbitrary textual description of the field.
    :param metadata: An optional key-value data.
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

    check_dispatch = _check_dispatch()
    for key in kwargs:
        if key not in check_dispatch:
            raise SchemaInitError(
                f"custom check '{key}' is not available. Make sure you use "
                "pandera.extensions.register_check_method decorator to "
                "register your custom check method."
            )

    for arg_name, check_constructor in check_dispatch.items():
        arg_value = args.get(arg_name, kwargs.get(arg_name))
        if arg_value is None:
            continue
        if isinstance(arg_value, dict):
            check_ = check_constructor(**arg_value, **check_kwargs)
        else:
            check_ = check_constructor(arg_value, **check_kwargs)
        checks.append(check_)
    if unique:
        raise SchemaInitError(
            "unique Field argument not yet implemented for pyspark"
        )

    return FieldInfo(
        checks=checks or None,
        nullable=nullable,
        unique=unique,
        coerce=coerce,
        regex=regex,
        check_name=check_name,
        alias=alias,
        title=title,
        description=description,
        dtype_kwargs=dtype_kwargs,
        metadata=metadata,
    )


def _check_dispatch():
    return {
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


class CheckInfo(BaseCheckInfo):  # pylint:disable=too-few-public-methods
    """Captures extra information about a Check."""

    ...  # pylint:disable=unnecessary-ellipsis


class FieldCheckInfo(CheckInfo):  # pylint:disable=too-few-public-methods
    """Captures extra information about a Check assigned to a field."""

    def __init__(
        self,
        fields: Set[Union[str, FieldInfo]],
        check_fn: AnyCallable,
        regex: bool = False,
        **check_kwargs: Any,
    ) -> None:
        super().__init__(check_fn, **check_kwargs)
        self.fields = fields
        self.regex = regex


def _to_function_and_classmethod(
    fn: Union[AnyCallable, classmethod],
) -> Tuple[AnyCallable, classmethod]:
    if isinstance(fn, classmethod):
        fn, method = fn.__func__, cast(classmethod, fn)
    else:
        method = classmethod(fn)
    return fn, method


ClassCheck = Callable[[Union[classmethod, AnyCallable]], classmethod]


def check(*fields, regex: bool = False, **check_kwargs) -> ClassCheck:
    """Decorator to make DataFrameModel method a column check function.

    *new in 0.16.0*

    This indicates that the decorated method should be used to validate a field
    (column). The method will be converted to a classmethod. Therefore
    its signature must start with `cls` followed by regular check arguments.
    See the :ref:`User Guide <schema-model-custom-check>` for more.

    :param _fn: Method to decorate.
    :param check_kwargs: Keywords arguments forwarded to Check.
    """

    def _wrapper(fn: Union[classmethod, AnyCallable]) -> classmethod:
        check_fn, check_method = _to_function_and_classmethod(fn)
        check_kwargs.setdefault("description", fn.__doc__)
        setattr(
            check_method,
            CHECK_KEY,
            FieldCheckInfo(set(fields), check_fn, regex, **check_kwargs),
        )
        return check_method

    return _wrapper


def dataframe_check(_fn=None, **check_kwargs) -> ClassCheck:
    """Decorator to make DataFrameModel method a dataframe-wide check function.

    *new in 0.16.0*

    Decorate a method on the DataFrameModel indicating that it should be used to
    validate the DataFrame. The method will be converted to a classmethod.
    Therefore its signature must start with `cls` followed by regular check
    arguments. See the :ref:`User Guide <schema-model-dataframe-check>` for
    more.

    :param check_kwargs: Keywords arguments forwarded to Check.
    """

    def _wrapper(fn: Union[classmethod, AnyCallable]) -> classmethod:
        check_fn, check_method = _to_function_and_classmethod(fn)
        check_kwargs.setdefault("description", fn.__doc__)
        setattr(
            check_method,
            DATAFRAME_CHECK_KEY,
            CheckInfo(check_fn, **check_kwargs),
        )
        return check_method

    if _fn:
        return _wrapper(_fn)  # type: ignore
    return _wrapper
