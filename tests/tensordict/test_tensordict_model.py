"""Unit tests for TensorDictModel."""

import pytest

try:
    import torch
    from tensordict import TensorDict

    from pandera.tensordict import DataType  # type: ignore[attr-defined]

    _DataType = DataType  # type: ignore[misc]
except ImportError:
    torch = None
    TensorDict = None
    DataType = None  # type: ignore[misc, assignment]

torch_condition = pytest.mark.skipif(
    torch is None, reason="torch not installed"
)
