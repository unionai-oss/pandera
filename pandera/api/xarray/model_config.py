"""Configuration objects for xarray declarative models."""

from typing import Any

from pandera.api.base.model_config import BaseModelConfig
from pandera.api.base.types import StrictType


class DataArrayConfig(BaseModelConfig):
    """Options for :class:`~pandera.api.xarray.model.DataArrayModel`.

    Schema-level defaults for the underlying :class:`DataArraySchema`.
    Field-level :func:`~pandera.api.xarray.model_components.Field` values on
    the ``data`` attribute override these when set.
    """

    dtype: Any | None = None
    dims: tuple[str | None, ...] | None = None
    sizes: dict[str, int | None] | None = None
    shape: tuple[int | None, ...] | None = None
    coerce: bool = False
    nullable: bool = False
    strict_coords: StrictType = False
    strict_attrs: StrictType = False
    attrs: dict[str, Any] | None = None
    chunked: bool | None = None
    array_type: Any | None = None


class DatasetConfig(BaseModelConfig):
    """Options for :class:`~pandera.api.xarray.model.DatasetModel`."""

    strict: StrictType = False
    strict_coords: StrictType = False
    strict_attrs: StrictType = False
    dims: tuple[str | None, ...] | None = None
    sizes: dict[str, int | None] | None = None
