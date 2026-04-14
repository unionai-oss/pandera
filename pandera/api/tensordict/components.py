"""Tensor component for TensorDict validation."""

from __future__ import annotations

from typing import Any

from pandera.api.base.types import CheckList
from pandera.api.dataframe.components import ComponentSchema

try:
    import torch
except ImportError:
    torch = None


class Tensor(ComponentSchema):
    """Schema component for validating a tensor in a TensorDict."""

    def __init__(
        self,
        dtype: Any = None,
        shape: tuple[int | None, ...] | None = None,
        checks: CheckList | None = None,
        nullable: bool = False,
        coerce: bool = False,
        name: str | None = None,
        title: str | None = None,
        description: str | None = None,
        metadata: dict | None = None,
        drop_invalid_rows: bool = False,
    ) -> None:
        """Create a tensor schema component.

        :param dtype: Expected torch dtype for the tensor.
        :param shape: Expected shape for the tensor. Use None for dimensions
            that can be any size. The first dimension should typically be None
            to allow for arbitrary batch sizes.
        :param checks: Checks to verify validity of the tensor values.
        :param nullable: Whether or not the tensor can be None.
        :param coerce: If True, coerce the tensor to the specified dtype.
        :param name: Name of the tensor in the TensorDict.
        :param title: A human-readable label for the tensor.
        :param description: An arbitrary textual description of the tensor.
        :param metadata: An optional key-value data.
        :param drop_invalid_rows: Not applicable for TensorDict.

        :raises SchemaInitError: if impossible to build schema from parameters
        """
        if dtype is not None and torch is not None:
            from pandera.engines import tensordict_engine
            dtype = tensordict_engine.Engine.dtype(dtype) if dtype else None

        super().__init__(
            dtype=dtype,
            checks=checks,
            nullable=nullable,
            coerce=coerce,
            name=name,
            title=title,
            description=description,
            metadata=metadata,
            drop_invalid_rows=drop_invalid_rows,
        )
        self.shape = shape

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, value) -> None:
        if torch is None:
            self._dtype = value
        else:
            from pandera.engines import tensordict_engine
            self._dtype = tensordict_engine.Engine.dtype(value) if value else None

    def __repr__(self) -> str:
        dtype_str = self._dtype.type if hasattr(self._dtype, 'type') else str(self._dtype)
        return (
            f"Tensor(dtype={dtype_str}, shape={self.shape}, "
            f"nullable={self.nullable})"
        )
