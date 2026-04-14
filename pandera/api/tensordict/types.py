"""Type definitions for pandera tensordict integration."""

from typing import TYPE_CING, Any, Union

import torch

from pandera.api.checks import Check

if TYPE_CING:
    from tensordict import TensorDict, tensorclass

TensorDictCheckObjects = Any

TensorDictInputType = Union["TensorDict", "tensorclass", torch.Tensor]

TensorDtypeInputTypes = Union[torch.dtype, str, type]


def is_tensordict(obj: Any) -> bool:
    """Check if object is a TensorDict or tensorclass."""
    try:
        from tensordict import TensorDict, tensorclass

        return isinstance(obj, (TensorDict, tensorclass))
    except ImportError:
        return False
