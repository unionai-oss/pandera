"""DataFrameModel components"""

from collections.abc import Callable, Iterable
from typing import (
    Any,
    Union,
    cast,
)

from pandera.api.base.model_components import (
    BaseCheckInfo,
    BaseFieldInfo,
    BaseParserInfo,
    CheckArg,
    ParserArg,
    to_checklist,
    to_parserlist,
)
from pandera.api.checks import Check
from pandera.errors import SchemaInitError

AnyCallable = Callable[..., Any]

CHECK_KEY = "__check_config__"
DATAFRAME_CHECK_KEY = "__dataframe_check_config__"
PARSER_KEY = "__parser_config__"
DATAFRAME_PARSER_KEY = "__dataframe_parser_config__"


class FieldInfo(BaseFieldInfo):
    """Captures extra information about a field.

    *new in 0.5.0*
    """

    def _get_schema_properties(
        self,
        dtype: Any,
        checks: CheckArg | None = None,
        parsers: ParserArg | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if self.dtype_kwargs:
            dtype = dtype(**self.dtype_kwargs)  # type: ignore
        return {
            "dtype": dtype,
            "checks": self.checks + to_checklist(checks),
            "parsers": self.parses + to_parserlist(parsers),
            **kwargs,
        }

    def column_properties(
        self,
        dtype: Any,
        checks: CheckArg | None = None,
        parsers: ParserArg | None = None,
        required: bool = True,
        name: str | None = None,
    ) -> dict[str, Any]:
        """Create a schema_components.Column from a field."""
        return self._get_schema_properties(
            dtype,
            nullable=self.nullable,
            unique=self.unique,
            coerce=self.coerce,
            regex=self.regex,
            required=required,
            name=name,
            checks=checks,
            parsers=parsers,
            title=self.title,
            description=self.description,
            default=self.default,
            metadata=self.metadata,
        )

    def index_properties(
        self,
        dtype: Any,
        checks: CheckArg | None = None,
        name: str | None = None,
    ) -> dict[str, Any]:
        """Create a schema_components.Index from a field."""
        return self._get_schema_properties(
            dtype,
            nullable=self.nullable,
            unique=self.unique,
            coerce=self.coerce,
            name=name,
            checks=checks,
            title=self.title,
            description=self.description,
            default=self.default,
        )

    @property
    def properties(self) -> dict[str, Any]:
        """Get column properties."""
        return {
            "dtype": self.dtype_kwargs,
            "checks": self.checks,
            "parses": self.parses,
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
    eq: Any | None = None,
    ne: Any | None = None,
    gt: Any | None = None,
    ge: Any | None = None,
    lt: Any | None = None,
    le: Any | None = None,
    in_range: Union[
        tuple[Any, Any],
        tuple[Any, Any, bool, bool],
        tuple[Any, Any, bool, bool, bool],
        tuple[Any, Any, bool, bool, bool, bool],
        dict[str, Any],
        None,
    ] = None,
    isin: Iterable[Any] | None = None,
    notin: Iterable[Any] | None = None,
    str_contains: str | None = None,
    str_endswith: str | None = None,
    str_length: Union[
        int,
        tuple[int],
        tuple[int, int],
        dict[str, int],
        None,
    ] = None,
    str_matches: str | None = None,
    str_startswith: str | None = None,
    nullable: bool = False,
    unique: bool = False,
    coerce: bool = False,
    regex: bool = False,
    ignore_na: bool = True,
    raise_warning: bool = False,
    n_failure_cases: int | None = None,
    alias: Any | None = None,
    check_name: bool | None = None,
    dtype_kwargs: dict[str, Any] | None = None,
    title: str | None = None,
    description: str | None = None,
    default: Any | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Column or index field specification of a DataFrameModel.

    *new in 0.5.0*

    Some arguments apply only to numeric dtypes and some apply only to ``str``.
    See the :ref:`User Guide <dataframe-models>` for more information.

    The keyword-only arguments for argument names ``eq`` to ``str_startswith`` are dispatched
    to the built-in :py:class:`~pandera.api.checks.Check` methods if the value
    is a dictionary. If the value is a tuple, it is unpacked as positional arguments.

    :param eq: Check that the column/index is equal to a value.
        See :func:`~pandera.api.checks.Check.equal_to` for more information.
    :param ne: Check that the column/index is not equal to a value.
        See :func:`~pandera.api.checks.Check.not_equal_to` for more information.
    :param gt: Check that the column/index is greater than a value.
        See :func:`~pandera.api.checks.Check.greater_than` for more information.
    :param ge: Check that the column/index is greater than or equal to a value.
        See :func:`~pandera.api.checks.Check.greater_than_or_equal_to` for more information.
    :param lt: Check that the column/index is less than a value.
        See :func:`~pandera.api.checks.Check.less_than` for more information.
    :param le: Check that the column/index is less than or equal to a value.
        See :func:`~pandera.api.checks.Check.less_than_or_equal_to` for more information.
    :param in_range: Check that the column/index is within a range.
        See :func:`~pandera.api.checks.Check.in_range` for more information.
    :param isin: Check that the column/index is in a set of values.
        See :func:`~pandera.api.checks.Check.isin` for more information.
    :param notin: Check that the column/index is not in a set of values.
        See :func:`~pandera.api.checks.Check.notin` for more information.
    :param str_contains: Check that the column/index contains a substring.
        See :func:`~pandera.api.checks.Check.str_contains` for more information.
    :param str_endswith: Check that the column/index ends with a substring.
        See :func:`~pandera.api.checks.Check.str_endswith` for more information.
    :param str_length: Check that the length of the column/index is within a range.
        See :func:`~pandera.api.checks.Check.str_length` for more information.
    :param str_matches: Check that the column/index matches a regex pattern.
        See :func:`~pandera.api.checks.Check.str_matches` for more information.
    :param str_startswith: Check that the column/index starts with a substring.
        See :func:`~pandera.api.checks.Check.str_startswith` for more information.
    :param nullable: Whether or not the column/index can contain null values.
    :param unique: Whether column values should be unique.
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
    :param default: Optional default value of the field.
    :param metadata: An optional key-value data.
    :param kwargs: Specify custom checks that have been registered with the
        :class:`~pandera.extensions.register_check_method` decorator.
    """
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
        elif isinstance(arg_value, tuple):
            # Unpack tuples as positional args (e.g., str_length=(1, 5))
            check_ = check_constructor(*arg_value, **check_kwargs)
        else:
            check_ = check_constructor(arg_value, **check_kwargs)
        checks.append(check_)

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
        default=default,
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
        "between": Check.between,
        "isin": Check.isin,
        "notin": Check.notin,
        "str_contains": Check.str_contains,
        "str_endswith": Check.str_endswith,
        "str_matches": Check.str_matches,
        "str_length": Check.str_length,
        "str_startswith": Check.str_startswith,
        "unique_values_eq": Check.unique_values_eq,
        **Check.REGISTERED_CUSTOM_CHECKS,
    }


class CheckInfo(BaseCheckInfo):
    """Captures extra information about a Check."""


class FieldCheckInfo(CheckInfo):
    """Captures extra information about a Check assigned to a field."""

    def __init__(
        self,
        fields: set[Union[str, FieldInfo]],
        check_fn: AnyCallable,
        regex: bool = False,
        **check_kwargs: Any,
    ) -> None:
        super().__init__(check_fn, **check_kwargs)
        self.fields = fields
        self.regex = regex


class ParserInfo(BaseParserInfo):
    """Captures extra information about a Parser."""


class FieldParserInfo(ParserInfo):
    """Captures extra information about a Parser assigned to a field."""

    def __init__(
        self,
        fields: set[Union[str, FieldInfo]],
        parser_fn: AnyCallable,
        regex: bool = False,
        **parser_kwargs: Any,
    ) -> None:
        super().__init__(parser_fn, **parser_kwargs)
        self.fields = fields
        self.regex = regex


def _to_function_and_classmethod(
    fn: Union[AnyCallable, classmethod],
) -> tuple[AnyCallable, classmethod]:
    if isinstance(fn, classmethod):
        fn, method = fn.__func__, cast(classmethod, fn)
    else:
        method = classmethod(fn)
    return fn, method


ClassCheck = Callable[[Union[classmethod, AnyCallable]], classmethod]


def check(*fields, regex: bool = False, **check_kwargs) -> ClassCheck:
    """Defines DataFrameModel check methods for columns/indexes.

    *new in 0.5.0*

    This indicates that the decorated method should be used to validate a field
    (column or index). The method will be converted to a classmethod. Therefore
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
    """Defines DataFrameModel check methods for dataframes.

    *new in 0.5.0*

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


ClassParser = Callable[[Union[classmethod, AnyCallable]], classmethod]


def parser(*fields, **parser_kwargs) -> ClassParser:
    """Defines DataFrameModel parse methods for columns/indexes."""

    def _wrapper(fn: Union[classmethod, AnyCallable]) -> classmethod:
        parser_fn, parser_method = _to_function_and_classmethod(fn)
        parser_kwargs.setdefault("description", fn.__doc__)
        setattr(
            parser_method,
            PARSER_KEY,
            FieldParserInfo(set(fields), parser_fn, **parser_kwargs),
        )
        return parser_method

    return _wrapper


def dataframe_parser(_fn=None, **parser_kwargs) -> ClassParser:
    """Defines DataFrameModel parse methods for dataframes."""

    def _wrapper(fn: Union[classmethod, AnyCallable]) -> classmethod:
        parser_fn, parser_method = _to_function_and_classmethod(fn)
        parser_kwargs.setdefault("description", fn.__doc__)
        setattr(
            parser_method,
            DATAFRAME_PARSER_KEY,
            ParserInfo(parser_fn, **parser_kwargs),
        )
        return parser_method

    return _wrapper(_fn)  # type: ignore
