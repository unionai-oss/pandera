"""Pandera TensorDict API."""

from __future__ import annotations

# mypy: disable-error-code=attr-defined
from typing import TYPE_CHECKING

from pandera import errors

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
]
