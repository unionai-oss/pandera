"""Type definitions for pandera tensordict integration."""

from __future__ import annotations

from typing import Any


def is_tensordict(obj: Any) -> bool:
    """Check if object is a TensorDict (any TensorDictBase subclass) or a
    tensorclass instance."""
    try:
        from tensordict import is_tensor_collection

        return is_tensor_collection(obj)
    except ImportError:
        return False
