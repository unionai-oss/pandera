"""Field descriptors for TensorDict declarative models."""

from collections.abc import Iterable
from typing import Any, Union

from pandera.api.base.model_components import to_checklist
from pandera.api.checks import Check
from pandera.api.dataframe.model_components import FieldInfo, _check_dispatch
from pandera.errors import SchemaInitError

CHECK_KEY = "__check_config__"


class TensorDictFieldInfo(FieldInfo):
    """Field metadata for :class:`TensorDictModel`."""

    def __init__(
        self,
        *,
        shape: tuple[int | None, ...] | None = None,
        required: bool = True,
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
        self.shape = shape
        self.required = required

    def to_tensor_kwargs(
        self,
        dtype: Any,
        *,
        optional: bool,
        checks: Any = None,
    ) -> dict[str, Any]:
        """Keyword args for :class:`~pandera.api.tensordict.components.Tensor`."""
        if self.dtype_kwargs:
            dtype = dtype(**self.dtype_kwargs)
        return {
            "dtype": dtype,
            "checks": to_checklist(self.checks) + to_checklist(checks),
            "shape": self.shape,
            "name": self.alias,
            "title": self.title,
            "description": self.description,
            "nullable": self.nullable,
            "coerce": self.coerce,
        }


def Field(
    *,
    dtype: Any = None,
    shape: tuple[int | None, ...] | None = None,
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
    no_nan: bool = False,
    no_inf: bool = False,
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
    **kwargs: Any,
) -> Any:
    """Field specification for TensorDict models."""
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

    return TensorDictFieldInfo(
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
        shape=shape,
    )
