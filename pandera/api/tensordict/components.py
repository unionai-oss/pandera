"""Tensor component for TensorDict schemas."""

from __future__ import annotations

from typing import Any, Union

from pandera.api.base.types import CheckList
from pandera.engines import tensordict_engine

#: A tensor key in a :class:`~pandera.api.tensordict.container.TensorDictSchema`.
#: May be a plain key (e.g. ``"observation"``) or a nested tuple key (e.g.
#: ``("next", "observation")``) matching TorchRL's nested TensorDict layout.
TensorDictKey = Union[str, tuple[str, ...]]


class Tensor:
    """Validate a single tensor entry in a TensorDict.

    Analogous to :class:`~pandera.api.pandas.components.Column` for DataFrames
    and :class:`~pandera.api.xarray.components.DataVar` for xarray Datasets.
    """

    def __init__(
        self,
        dtype: Any = None,
        shape: tuple[int | None, ...] | None = None,
        checks: CheckList | None = None,
        nullable: bool = False,
        coerce: bool = False,
        name: TensorDictKey | None = None,
        required: bool = True,
    ) -> None:
        """Create tensor validator.

        :param dtype: PyTorch dtype (e.g., ``torch.float32``, ``torch.int64``).
            If a string is provided, it will be converted to the corresponding
            torch.dtype.
        :param shape: Expected shape tuple. Use ``None`` for flexible dimensions.
            For example, ``(None, 10)`` allows any batch size with 10 features.
        :param checks: List of :class:`~pandera.api.checks.Check` instances for
            value validation.
        :param nullable: Whether the key can be missing from the TensorDict.
        :param coerce: Whether to coerce dtype during validation.
        :param name: Tensor key name in TensorDict. If ``None``, uses the key
            from the schema definition. May be a nested tuple key.
        :param required: Whether the tensor is required (always ``True`` for
            now, as missing keys are handled separately).
        """
        self.dtype = (
            tensordict_engine.Engine.dtype(dtype)
            if dtype is not None
            else None
        )
        self.shape = shape

        # Normalize checks to a list
        self.checks: list = []
        if checks is not None:
            if isinstance(checks, (list, tuple)):
                self.checks = list(checks)
            else:
                self.checks = [checks]

        self.nullable = nullable
        self.coerce = coerce
        self.name = name
        self.required = required

    def __repr__(self) -> str:
        return f"Tensor(dtype={self.dtype}, shape={self.shape})"
