"""Xarray container schemas."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union

from pandera.api.base.types import CheckList, ParserList, StrictType
from pandera.api.checks import Check
from pandera.api.hypotheses import Hypothesis
from pandera.api.parsers import Parser
from pandera.api.xarray.base import BaseDataArraySchema as _BaseDataArraySchema
from pandera.api.xarray.base import BaseDatasetSchema as _BaseDatasetSchema
from pandera.api.xarray.utils import get_validation_depth
from pandera.config import config_context, get_config_context
from pandera.errors import BackendNotFoundError, SchemaDefinitionError

if TYPE_CHECKING:
    from pandera.api.xarray.components import DataVar


import xarray as xr


def _normalize_dims(
    dims: Union[tuple[str | None, ...], list[str | None], dict[str, str]] | None,
):
    if dims is None:
        return None
    if isinstance(dims, dict):
        return tuple(dims.keys())
    if isinstance(dims, list):
        return tuple(dims)
    return dims


class DataArraySchema(_BaseDataArraySchema):
    """A lightweight xarray DataArray validator."""

    def __init__(
        self,
        dtype: Any | None = None,
        dims: Union[tuple[str | None, ...], list[str | None], dict[str, str]] | None = None,
        sizes: dict[str, int | None] | None = None,
        shape: tuple[int | None, ...] | None = None,
        coords: dict[str, Any] | None = None,
        attrs: dict[str, Any] | None = None,
        name: str | None = None,
        checks: CheckList | None = None,
        parsers: ParserList | None = None,
        coerce: bool = False,
        nullable: bool = False,
        chunked: bool | None = None,
        array_type: Any | None = None,
        strict_coords: StrictType = False,
        strict_attrs: StrictType = False,
        title: str | None = None,
        description: str | None = None,
        metadata: dict | None = None,
    ):
        """Initialize a DataArraySchema.

        :param dtype: datatype of the DataArray.
        :param dims: dimension names. Can be a list of dimension names or a
            dict mapping dimension names to dimension types.
        :param sizes: size requirements for dimensions.
        :param shape: shape requirements for the DataArray.
        :param coords: coordinate specifications.
        :param attrs: attribute specifications.
        :param name: name of the DataArray.
        :param checks: checks applied to the whole DataArray (after structure).
        :param parsers: parsers applied to the whole DataArray before checks.
        :param coerce: whether or not to coerce data.
        :param nullable: whether the DataArray can contain null values.
        :param chunked: if True, require a Dask-backed array; if False,
            require eager data; if None, do not check.
        :param array_type: expected type of underlying array (e.g. ``numpy.ndarray``).
        :param strict_coords: whether to enforce strict coordinate validation.
        :param strict_attrs: whether to enforce strict attribute validation.
        :param title: A human-readable label for the schema.
        :param description: An arbitrary textual description of the schema.
        :param metadata: An optional key-value data.
        """
        if checks is None:
            checks = []
        if isinstance(checks, (Check, Hypothesis)):
            checks = [checks]

        if parsers is None:
            parsers = []
        if isinstance(parsers, Parser):
            parsers = [parsers]
        if sizes is not None and shape is not None:
            raise SchemaDefinitionError(
                "Pass only one of `sizes` and `shape` on DataArraySchema."
            )

        super().__init__(
            dtype=dtype,
            checks=checks,
            parsers=parsers,
            coerce=coerce,
            name=name,
            title=title,
            description=description,
            metadata=metadata,
        )

        self.dims = _normalize_dims(dims)
        self.sizes = sizes
        self.shape = shape
        self.coords = coords
        self.attrs = attrs
        self.nullable = nullable
        self.chunked = chunked
        self.array_type = array_type
        self.strict_coords = strict_coords
        self.strict_attrs = strict_attrs

    def validate(
        self,
        check_obj: xr.DataArray,
        head: int | None = None,
        tail: int | None = None,
        sample: int | None = None,
        random_state: int | None = None,
        lazy: bool = False,
        inplace: bool = False,
    ) -> xr.DataArray:
        """Validate a DataArray based on the schema specification.

        :param check_obj: the DataArray to be validated.
        :param head: validate the first ``n`` positions along the **first**
            dimension only (see backend subsampling).
        :param tail: validate the last ``n`` positions along the first dimension.
        :param sample: random subset of size ``n`` along the first dimension.
        :param random_state: random seed for the ``sample`` argument.
        :param lazy: if True, lazily evaluates DataArray against all validation
            checks and raises a ``SchemaErrors``. Otherwise, raise
            ``SchemaError`` as soon as one occurs.
        :param inplace: if True, applies coercion to the object of validation,
            otherwise creates a copy of the data.
        :returns: validated ``DataArray``

        :raises SchemaError: when ``DataArray`` violates built-in or custom
            checks.

        Chunked (Dask-backed) arrays default to
        :attr:`~pandera.config.ValidationDepth.SCHEMA_ONLY` for data-level
        checks unless ``validation_depth`` is set (see
        :func:`pandera.api.xarray.utils.get_validation_depth`).
        """
        if not get_config_context().validation_enabled:
            return check_obj

        with config_context(validation_depth=get_validation_depth(check_obj)):
            return self._validate(
                check_obj=check_obj,
                head=head,
                tail=tail,
                sample=sample,
                random_state=random_state,
                lazy=lazy,
                inplace=inplace,
            )

    def _validate(
        self,
        check_obj: xr.DataArray,
        head: int | None = None,
        tail: int | None = None,
        sample: int | None = None,
        random_state: int | None = None,
        lazy: bool = False,
        inplace: bool = False,
    ) -> xr.DataArray:
        return self.get_backend(check_obj).validate(
            check_obj,
            schema=self,
            head=head,
            tail=tail,
            sample=sample,
            random_state=random_state,
            lazy=lazy,
            inplace=inplace,
        )

    @staticmethod
    def register_default_backends(check_obj_cls: type):
        from pandera.backends.xarray.register import register_xarray_backends

        _cls = check_obj_cls
        try:
            register_xarray_backends(f"{_cls.__module__}.{_cls.__name__}")
        except BackendNotFoundError:
            for base_cls in _cls.__bases__:
                base_cls_name = f"{base_cls.__module__}.{base_cls.__name__}"
                try:
                    register_xarray_backends(base_cls_name)
                except BackendNotFoundError:
                    pass


class DatasetSchema(_BaseDatasetSchema):
    """A lightweight xarray Dataset validator."""

    def __init__(
        self,
        data_vars: dict[str, Union[DataVar, DataArraySchema, None]] | None = None,
        coords: dict[str, Any] | None = None,
        dims: Union[tuple[str, ...], list[str], dict[str, str]] | None = None,
        sizes: dict[str, int | None] | None = None,
        attrs: dict[str, Any] | None = None,
        checks: CheckList | None = None,
        parsers: ParserList | None = None,
        strict: StrictType | str = False,
        strict_coords: StrictType = False,
        strict_attrs: StrictType = False,
        name: str | None = None,
        title: str | None = None,
        description: str | None = None,
        metadata: dict | None = None,
    ):
        """Initialize a DatasetSchema.

        :param data_vars: mapping of logical names to
            :class:`~pandera.api.xarray.components.DataVar`,
            :class:`DataArraySchema`, or ``None`` (variable must exist,
            no value checks).
        :param coords: coordinate specifications.
        :param dims: dimension names. Can be a list of dimension names or a
            dict mapping dimension names to dimension types.
        :param sizes: size requirements for dimensions.
        :param attrs: attribute specifications.
        :param checks: checks applied to the whole Dataset (after per-variable
            validation).
        :param parsers: parsers applied to the whole Dataset before checks.
        :param strict: whether to enforce strict validation.
        :param strict_coords: whether to enforce strict coordinate validation.
        :param strict_attrs: whether to enforce strict attribute validation.
        :param title: A human-readable label for the schema.
        :param description: An arbitrary textual description of the schema.
        :param metadata: An optional key-value data.
        """
        if data_vars is None:
            data_vars = {}
        from pandera.api.xarray.components import DataVar as _DataVar

        for _key, spec in data_vars.items():
            if spec is None:
                continue
            if not isinstance(spec, (_DataVar, DataArraySchema)):
                raise SchemaDefinitionError(
                    "DatasetSchema data_vars values must be DataVar, "
                    "DataArraySchema, or None."
                )

        if checks is None:
            checks = []
        if isinstance(checks, (Check, Hypothesis)):
            checks = [checks]

        if parsers is None:
            parsers = []
        if isinstance(parsers, Parser):
            parsers = [parsers]

        super().__init__(
            dtype=None,
            checks=checks,
            parsers=parsers,
            coerce=False,
            name=name,
            title=title,
            description=description,
            metadata=metadata,
        )

        self.data_vars = data_vars
        self.coords = coords
        self.dims = _normalize_dims(dims)
        self.sizes = sizes
        self.attrs = attrs
        self.strict = strict
        self.strict_coords = strict_coords
        self.strict_attrs = strict_attrs

    def validate(
        self,
        check_obj: xr.Dataset,
        head: int | None = None,
        tail: int | None = None,
        sample: int | None = None,
        random_state: int | None = None,
        lazy: bool = False,
        inplace: bool = False,
    ) -> xr.Dataset:
        """Validate a Dataset based on the schema specification.

        :param check_obj: the Dataset to be validated.
        :param head: validate the first ``n`` positions along the **first**
            dimension only (see backend subsampling).
        :param tail: validate the last ``n`` positions along the first dimension.
        :param sample: random subset of size ``n`` along the first dimension.
        :param random_state: random seed for the ``sample`` argument.
        :param lazy: if True, lazily evaluates Dataset against all validation
            checks and raises a ``SchemaErrors``. Otherwise, raise
            ``SchemaError`` as soon as one occurs.
        :param inplace: if True, applies coercion to the object of validation,
            otherwise creates a copy of the data.
        :returns: validated ``Dataset``

        :raises SchemaError: when ``Dataset`` violates built-in or custom
            checks.

        If any data variable is chunked (Dask-backed), data-level checks default
        to :attr:`~pandera.config.ValidationDepth.SCHEMA_ONLY` unless
        ``validation_depth`` is configured; see
        :func:`pandera.api.xarray.utils.get_validation_depth`.
        """
        if not get_config_context().validation_enabled:
            return check_obj

        with config_context(validation_depth=get_validation_depth(check_obj)):
            return self._validate(
                check_obj=check_obj,
                head=head,
                tail=tail,
                sample=sample,
                random_state=random_state,
                lazy=lazy,
                inplace=inplace,
            )

    def _validate(
        self,
        check_obj: xr.Dataset,
        head: int | None = None,
        tail: int | None = None,
        sample: int | None = None,
        random_state: int | None = None,
        lazy: bool = False,
        inplace: bool = False,
    ) -> xr.Dataset:
        return self.get_backend(check_obj).validate(
            check_obj,
            schema=self,
            head=head,
            tail=tail,
            sample=sample,
            random_state=random_state,
            lazy=lazy,
            inplace=inplace,
        )

    @staticmethod
    def register_default_backends(check_obj_cls: type):
        from pandera.backends.xarray.register import register_xarray_backends

        _cls = check_obj_cls
        try:
            register_xarray_backends(f"{_cls.__module__}.{_cls.__name__}")
        except BackendNotFoundError:
            for base_cls in _cls.__bases__:
                base_cls_name = f"{base_cls.__module__}.{base_cls.__name__}"
                try:
                    register_xarray_backends(base_cls_name)
                except BackendNotFoundError:
                    pass
