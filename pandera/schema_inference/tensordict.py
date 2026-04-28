"""Module for inferring TensorDict schema from data."""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

import torch

from pandera.api.tensordict.components import Tensor
from pandera.api.tensordict.container import TensorDictSchema


@overload
def infer_schema(
    tensordict: torch.TensorDict,
) -> TensorDictSchema: ...
@overload
def infer_schema(
    tensordict: Any,
) -> TensorDictSchema: ...


def infer_schema(tensordict: Any):
    """Infer schema for a TensorDict or tensorclass object.

    Automatically detects dtypes, shapes, and value statistics from data.

    :param tensordict: TensorDict or tensorclass object to infer.
    :returns: TensorDictSchema with inferred properties.
    :raises TypeError: if tensordict is not a valid TensorDict/tensorclass.

    Example:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> import pandera.tensordict as pa
        >>> td = TensorDict({
        ...     "obs": torch.randn(32, 10),
        ...     "action": torch.randint(0, 4, (32,))
        ... }, batch_size=[32])
        >>> schema = pa.infer_schema(td)
    """
    try:
        from tensordict import TensorDict as TD
    except ImportError as exc:
        raise ImportError(
            "tensordict is required for infer_schema but is not installed. "
            "Install with: pip install tensordict"
        ) from exc

    if not _is_tensordict_instance(tensordict):
        raise TypeError(
            f"Expected TensorDict or tensorclass, got {type(tensordict)}"
        )

    keys = {}
    for key in _get_keys_from_tensordict(tensordict):
        tensor = tensordict.get(key)
        if tensor is not None:
            keys[key] = _infer_tensor_schema(tensor)

    # Convert batch_size from torch.Size to tuple
    batch_size_tuple = (
        tuple(tensordict.batch_size) if tensordict.batch_size else None
    )

    return TensorDictSchema(
        keys=keys,
        batch_size=batch_size_tuple,
        coerce=True,
    )


def _get_keys_from_tensordict(obj):
    """Get keys from a TensorDict or tensorclass object."""
    try:
        # For TensorDict, use .keys()
        if hasattr(obj, "keys") and callable(getattr(obj, "keys")):
            return list(obj.keys())

        # For tensorclass, try to get from _tensordict
        if hasattr(obj, "_tensordict"):
            td = obj._tensordict
            if hasattr(td, "keys") and callable(getattr(td, "keys")):
                return list(td.keys())
    except Exception:
        pass

    # Fallback: try to iterate over attributes
    keys = []
    for attr in dir(obj):
        if not attr.startswith("_"):
            val = getattr(obj, attr, None)
            if isinstance(val, torch.Tensor):
                keys.append(attr)

    return keys


def _is_tensordict_instance(obj) -> bool:
    """Check if object is a TensorDict or tensorclass instance."""
    try:
        from tensordict import TensorDict as TD

        # Check for regular TensorDict
        if isinstance(obj, TD):
            return True
        # Check for tensorclass (has _is_tensorclass attribute)
        if hasattr(obj, "_is_tensorclass") and obj._is_tensorclass:
            return True
        return False
    except ImportError:
        return False


def _infer_tensor_schema(tensor: torch.Tensor) -> Tensor:
    """Infer Tensor schema from a PyTorch tensor.

    :param tensor: PyTorch tensor to infer.
    :returns: Tensor component with inferred dtype, shape, and checks.
    """
    # Infer dtype
    dtype = tensor.dtype

    # Infer shape (None for dynamic dimensions)
    shape = tuple(s if s > 0 else None for s in tensor.shape)

    # Infer value range statistics
    checks = _infer_tensor_checks(tensor)

    return Tensor(dtype=dtype, shape=shape, checks=checks)


def _infer_tensor_checks(tensor: torch.Tensor) -> list | None:
    """Infer check constraints from tensor data.

    :param tensor: PyTorch tensor to analyze.
    :returns: List of Check objects or None if no meaningful checks can be inferred.
    """
    import pandera as pa

    # Only infer checks for numeric tensors
    if not torch.is_floating_point(tensor) and not torch.is_complex(tensor):
        return []

    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()

    checks = []

    if not has_nan and not has_inf:
        try:
            min_val = tensor.min()
            max_val = tensor.max()

            # Use torch.isfinite on the tensor values
            if torch.all(torch.isfinite(min_val)) and torch.all(
                torch.isfinite(max_val)
            ):
                checks.append(
                    pa.Check.greater_than_or_equal_to(min_val.item())
                )
                checks.append(pa.Check.less_than_or_equal_to(max_val.item()))
        except (AttributeError, RuntimeError):
            pass

    return checks if checks else None
