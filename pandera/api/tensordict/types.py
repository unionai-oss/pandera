"""Type definitions for pandera tensordict integration."""

from typing import TYPE_CHECKING, Any, Union

from pandera.api.checks import Check

try:
    import torch
except ImportError:
    torch = None

if TYPE_CHECKING:
    from tensordict import TensorDict, tensorclass

TensorDictCheckObjects = Any

TensorDictInputType = Union["TensorDict", "tensorclass", torch.Tensor]  # type: ignore[union-attr]

TensorDtypeInputTypes = Union[torch.dtype, str, type]  # type: ignore[union-attr]


def is_tensordict(obj: Any) -> bool:
    """Check if object is a TensorDict or tensorclass."""
    try:
        from tensordict import TensorDict, tensorclass

        return isinstance(obj, (TensorDict, tensorclass))
    except ImportError:
        return False
