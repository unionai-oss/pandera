"""Pandera TensorDict API."""

from pandera import errors
from pandera.api.tensordict.components import Tensor
from pandera.api.tensordict.container import TensorDictSchema
from pandera.api.tensordict.model import TensorDictModel
from pandera.api.tensordict.model_components import Field

try:
    from pandera.engines.tensordict_engine import DataType
except ImportError:
    DataType = None

__all__ = [
    "Tensor",
    "TensorDictSchema",
    "TensorDictModel",
    "Field",
    "errors",
    "DataType",
]
