"""DataFrameModel components"""

from collections.abc import Callable, Iterable
from typing import (
    Any,
    Optional,
    TypeVar,
)

from pandera.api.dataframe.model_components import Field as _Field
from pandera.api.dataframe.components import ComponentSchema
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
    in_range: dict[str, Any] = None,
    isin: Iterable = None,
    notin: Iterable = None,
    str_contains: str | None = None,
    str_endswith: str | None = None,
    str_length: int | dict[str, Any] | None = None,
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
