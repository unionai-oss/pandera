"""DataFrameModel components"""

from collections.abc import Callable, Iterable
from typing import (
    Any,
    TypeVar,
    Union,
)

from pandera.api.dataframe.components import ComponentSchema
from pandera.api.dataframe.model_components import Field as _Field
from pandera.errors import SchemaInitError

AnyCallable = Callable[..., Any]
SchemaComponent = TypeVar("SchemaComponent", bound=ComponentSchema)


def Field(
    *,
    eq: Any = None,
    ne: Any = None,
    gt: Any = None,
    ge: Any = None,
    lt: Any = None,
    le: Any = None,
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
    alias: Any = None,
    check_name: bool | None = None,
    dtype_kwargs: dict[str, Any] | None = None,
    title: str | None = None,
    description: str | None = None,
    metadata: dict | None = None,
    **kwargs,
) -> Any:
    """Used to provide extra information about a field of a DataFrameModel.

    *new in 0.16.0*

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
    if unique:
        raise SchemaInitError(
            "unique Field argument not yet implemented for pyspark"
        )

    return _Field(
        eq=eq,
        ne=ne,
        gt=gt,
        ge=ge,
        lt=lt,
        le=le,
        in_range=in_range,
        isin=isin,
        notin=notin,
        str_contains=str_contains,
        str_endswith=str_endswith,
        str_length=str_length,
        str_matches=str_matches,
        str_startswith=str_startswith,
        nullable=nullable,
        unique=unique,
        coerce=coerce,
        regex=regex,
        ignore_na=ignore_na,
        raise_warning=raise_warning,
        n_failure_cases=n_failure_cases,
        alias=alias,
        check_name=check_name,
        dtype_kwargs=dtype_kwargs,
        title=title,
        description=description,
        metadata=metadata,
        **kwargs,
    )
