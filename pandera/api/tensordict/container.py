"""TensorDict Schema for pandera."""

from __future__ import annotations

import sys
from typing import Any

from pandera.api.base.types import CheckList
from pandera.api.dataframe.container import DataFrameSchema as _DataFrameSchema
from pandera.api.tensordict.components import Tensor
from pandera.api.tensordict.types import TensorDictCheckObjects
from pandera.config import config_context, get_config_context

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

try:
    import torch
except ImportError:
    torch = None


class TensorDictSchema(_DataFrameSchema[TensorDictCheckObjects]):
    """A schema for validating TensorDict and tensorclass objects."""

    def __init__(
        self,
        keys: dict[str, Tensor] | list[str] | None = None,
        batch_size: tuple[int, ...] | None = None,
        dtype: Any = None,
        checks: CheckList | None = None,
        coerce: bool = False,
        nullable: bool = False,
        name: str | None = None,
        title: str | None = None,
        description: str | None = None,
        metadata: dict | None = None,
        drop_invalid_rows: bool = False,
    ) -> None:
        """Create a TensorDict schema.

        :param keys: A dictionary mapping key names to Tensor schema objects,
            or a list of key names (all tensors will be validated for dtype only).
        :param batch_size: The expected batch size. Use None in dimensions
            to allow for any size.
        :param dtype: Expected dtype for all tensors in the container.
        :param checks: Checks to apply to the entire TensorDict.
        :param coerce: If True, coerce tensors to the specified dtype.
        :param nullable: Whether or not tensors can be None.
        :param name: Name of the schema.
        :param title: A human-readable label for the schema.
        :param description: An arbitrary textual description of the schema.
        :param metadata: An optional key-value data.
        :param drop_invalid_rows: Not applicable for TensorDict.

        :raises SchemaInitError: if impossible to build schema from parameters
        """
        if keys is None:
            keys = {}
        if isinstance(keys, list):
            keys = {k: Tensor(dtype=dtype) for k in keys}

        super().__init__(
            columns=keys,
            checks=checks,
            coerce=coerce,
            name=name,
            title=title,
            description=description,
            metadata=metadata,
            drop_invalid_rows=drop_invalid_rows,
        )
        self.batch_size = batch_size
        self._dtype = dtype

    @staticmethod
    def register_default_backends(check_obj_cls: type):
        from pandera.backends.tensordict.register import (
            register_tensordict_backends,
        )

        register_tensordict_backends()

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, value) -> None:
        if torch is None:
            self._dtype = value
        else:
            from pandera.engines import tensordict_engine

            self._dtype = (
                tensordict_engine.Engine.dtype(value) if value else None
            )

    def validate(
        self,
        check_obj: TensorDictCheckObjects,
        lazy: bool = False,
        inplace: bool = False,
    ) -> TensorDictCheckObjects:
        """Validate a TensorDict or tensorclass against the schema.

        :param check_obj: TensorDict or tensorclass to validate.
        :param lazy: If True, collect all errors. Otherwise, fail fast.
        :param inplace: If True, modify the input in place.
        :returns: Validated TensorDict or tensorclass.

        :raises SchemaError: when TensorDict violates schema.
        """
        if not get_config_context().validation_enabled:
            return check_obj

        with config_context(validation_depth=None):
            output = self.get_backend(check_obj).validate(
                check_obj=check_obj,
                schema=self,
                lazy=lazy,
                inplace=inplace,
            )

        return output

    @_DataFrameSchema.dtype.setter  # type: ignore[attr-defined]
    def dtype(self, value) -> None:
        """Set the dtype property."""
        if torch is None:
            self._dtype = value
        else:
            from pandera.engines import tensordict_engine

            self._dtype = (
                tensordict_engine.Engine.dtype(value) if value else None
            )

    def __repr__(self) -> str:
        return (
            f"TensorDictSchema(keys={list(self.columns.keys())}, "
            f"batch_size={self.batch_size})"
        )
