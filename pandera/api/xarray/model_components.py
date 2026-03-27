"""Field descriptors for xarray declarative models."""

from collections.abc import Iterable
from typing import Any, Union

from pandera.api.checks import Check
from pandera.api.dataframe.model_components import FieldInfo, _check_dispatch
from pandera.errors import SchemaInitError

CHECK_KEY = "__check_config__"
DATASET_CHECK_KEY = "__dataframe_check_config__"
PARSER_KEY = "__parser_config__"
DATASET_PARSER_KEY = "__dataframe_parser_config__"


class XarrayFieldInfo(FieldInfo):
    """Field metadata for :class:`DataArrayModel` / :class:`DatasetModel`."""

    def __init__(
        self,
        *,
        dims: tuple[str, ...] | None = None,
        sizes: dict[str, int | None] | None = None,
        shape: tuple[int | None, ...] | None = None,
        required: bool = True,
        aligned_with: tuple[str, ...] | None = None,
        broadcastable_with: tuple[str, ...] | None = None,
        nested_data_array_model: type | None = None,
        checks: Any = None,
        parses: Any = None,
        nullable: bool = False,
        unique: bool = False,
        coerce: bool = False,
        regex: bool = False,
        alias: Any = None,
        check_name: bool | None = None,
        dtype_kwargs: dict[str, Any] | None = None,
        title: str | None = None,
        description: str | None = None,
        default: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            checks=checks,
            parses=parses,
            nullable=nullable,
            unique=unique,
            coerce=coerce,
            regex=regex,
            alias=alias,
            check_name=check_name,
            dtype_kwargs=dtype_kwargs,
            title=title,
            description=description,
            default=default,
            metadata=metadata,
        )
        self.dims = dims
        self.sizes = sizes
        self.shape = shape
        self.required = required
        self.aligned_with = aligned_with
        self.broadcastable_with = broadcastable_with
        self.nested_data_array_model = nested_data_array_model

    def to_data_var_kwargs(
        self,
        dtype: Any,
        *,
        optional: bool,
    ) -> dict[str, Any]:
        """Keyword args for :class:`~pandera.api.xarray.components.DataVar`."""
        req = self.required and not optional
        return self._get_schema_properties(
            dtype,
            required=req,
            dims=self.dims,
            sizes=self.sizes,
            shape=self.shape,
            aligned_with=self.aligned_with,
            broadcastable_with=self.broadcastable_with,
            alias=self.alias,
            title=self.title,
            description=self.description,
            default=self.default,
            metadata=self.metadata,
            nullable=self.nullable,
            coerce=self.coerce,
        )


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
    required: bool = True,
    dims: tuple[str, ...] | None = None,
    sizes: dict[str, int | None] | None = None,
    shape: tuple[int | None, ...] | None = None,
    aligned_with: tuple[str, ...] | None = None,
    broadcastable_with: tuple[str, ...] | None = None,
    **kwargs: Any,
) -> XarrayFieldInfo:
    """Field specification for xarray models (mirrors dataframe :func:`Field`)."""
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
            check_ = check_constructor(*arg_value, **check_kwargs)
        else:
            check_ = check_constructor(arg_value, **check_kwargs)
        checks.append(check_)

    return XarrayFieldInfo(
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
        required=required,
        dims=dims,
        sizes=sizes,
        shape=shape,
        aligned_with=aligned_with,
        broadcastable_with=broadcastable_with,
    )
