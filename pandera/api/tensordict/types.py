"""Type definitions for pandera tensordict integration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union

from pandera.api.checks import Check

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from tensordict import TensorDict, tensorclass


def is_tensordict(obj: Any) -> bool:
    """Check if object is a TensorDict or tensorclass."""
    try:
        from tensordict import TensorDict, tensorclass

        return isinstance(obj, (TensorDict, tensorclass))
    except ImportError:
        return False
