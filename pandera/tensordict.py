"""Pandera TensorDict API."""

from __future__ import annotations

# mypy: disable-error-code=attr-defined
from typing import TYPE_CHECKING

from pandera import errors

try:
    from pandera.engines.tensordict_engine import DataType as _DataType

    DataType: type[_DataType] | None = _DataType  # type: ignore[misc]
except ImportError:
    DataType = None  # type: ignore[misc, assignment]

if TYPE_CHECKING:
    from pandera.api.tensordict.components import Tensor
    from pandera.api.tensordict.container import TensorDictSchema
    from pandera.api.tensordict.model import TensorDictModel

from pandera.api.tensordict.model_components import Field

__all__ = [
    "Tensor",
    "TensorDictSchema",
    "TensorDictModel",
    "Field",
    "errors",
    "DataType",
]
