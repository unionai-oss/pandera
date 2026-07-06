"""Tests for TensorDict schema inference."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict, tensorclass

import pandera.tensordict as pa


@pytest.fixture
def sample_tensor_dict():
    """Create a sample TensorDict for testing."""
    return TensorDict(
        {
            "observation": torch.randn(32, 10),
            "action": torch.randint(0, 4, (32,)),
            "reward": torch.randn(32),
            "done": torch.zeros(32, dtype=torch.bool),
        },
        batch_size=[32],
    )


class TestSchemaInference:
    """Test schema inference functionality."""

    def test_infer_schema_basic(self, sample_tensor_dict):
        """Test basic schema inference from TensorDict."""
        schema = pa.infer_schema(sample_tensor_dict)

        assert isinstance(schema, pa.TensorDictSchema)
        assert len(schema.keys) == 4
        assert "observation" in schema.keys
        assert "action" in schema.keys
        assert "reward" in schema.keys
        assert "done" in schema.keys

    def test_infer_schema_batch_size(self, sample_tensor_dict):
        """Test that batch_size is inferred correctly."""
        schema = pa.infer_schema(sample_tensor_dict)

        # batch_size might be torch.Size or tuple, both are acceptable
        assert schema.batch_size is not None
        if hasattr(schema.batch_size, "__iter__"):
            assert tuple(schema.batch_size) == (32,)

    def test_infer_schema_dtype(self, sample_tensor_dict):
        """Test that dtypes are inferred correctly."""
        schema = pa.infer_schema(sample_tensor_dict)

        # Compare using string representation since dtype may be wrapped
        obs_dtype = str(schema.keys["observation"].dtype)
        assert "float32" in obs_dtype or "torch.float32" in obs_dtype

        action_dtype = str(schema.keys["action"].dtype)
        assert "int64" in action_dtype or "torch.int64" in action_dtype

    def test_infer_schema_shape(self, sample_tensor_dict):
        """Test that shapes are inferred correctly."""
        schema = pa.infer_schema(sample_tensor_dict)

        obs_shape = (
            tuple(schema.keys["observation"].shape)
            if schema.keys["observation"].shape
            else None
        )
        assert obs_shape == (32, 10)

    def test_infer_schema_with_tensorclass(self):
        """Test schema inference with tensorclass."""

        @tensorclass
        class Data:
            x: torch.Tensor
            y: torch.Tensor

        tc = Data(
            x=torch.randn(16, 8), y=torch.randint(0, 2, (16,)), batch_size=[16]
        )

        schema = pa.infer_schema(tc)

        assert isinstance(schema, pa.TensorDictSchema)
        assert len(schema.keys) == 2

    def test_infer_schema_value_ranges(self, sample_tensor_dict):
        """Test that value range checks are inferred."""
        schema = pa.infer_schema(sample_tensor_dict)

        # Check that numeric tensors have range checks
        obs_schema = schema.keys["observation"]
        assert len(obs_schema.checks) >= 2  # min and max bounds

    def test_infer_schema_invalid_type(self):
        """Test that invalid types raise TypeError."""
        with pytest.raises(TypeError, match="Expected TensorDict"):
            pa.infer_schema({"not": "a tensordict"})

    def test_infer_schema_boolean_dtype(self, sample_tensor_dict):
        """Test that boolean dtypes are inferred correctly."""
        schema = pa.infer_schema(sample_tensor_dict)

        done_dtype = str(schema.keys["done"].dtype)
        assert "bool" in done_dtype or "torch.bool" in done_dtype

    def test_infer_schema_integer_dtypes(self, sample_tensor_dict):
        """Test that integer dtypes are handled correctly."""
        schema = pa.infer_schema(sample_tensor_dict)

        action_dtype = str(schema.keys["action"].dtype)
        assert "int64" in action_dtype or "torch.int64" in action_dtype
