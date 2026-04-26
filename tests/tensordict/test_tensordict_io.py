"""Tests for TensorDict schema serialization/deserialization."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict, tensorclass

import pandera.tensordict as pa


@pytest.fixture
def sample_schema():
    """Create a sample schema for testing."""
    return pa.TensorDictSchema(
        keys={
            "observation": pa.Tensor(dtype=torch.float32, shape=(None, 10)),
            "action": pa.Tensor(dtype=torch.int64, shape=(None,)),
            "reward": pa.Tensor(dtype=torch.float32, shape=(None,)),
        },
        batch_size=(32,),
        coerce=True,
    )


@pytest.fixture
def sample_tensor_dict():
    """Create a sample TensorDict for testing."""
    return TensorDict(
        {
            "observation": torch.randn(32, 10),
            "action": torch.randint(0, 4, (32,)),
            "reward": torch.randn(32),
        },
        batch_size=[32],
    )


class TestSchemaSerialization:
    """Test schema serialization functionality."""

    def test_serialize_schema_to_dict(self, sample_schema):
        """Test serializing schema to dictionary."""
        from pandera.io.tensordict_io import serialize_schema

        serialized = serialize_schema(sample_schema)

        assert isinstance(serialized, dict)
        assert "schema_type" in serialized
        assert serialized["schema_type"] == "tensordict"

    def test_serialize_deserialize_roundtrip(self, sample_schema):
        """Test round-trip serialization/deserialization."""
        # Serialize to YAML string
        yaml_str = pa.to_yaml(sample_schema)

        # Deserialize from YAML string
        deserialized = pa.from_yaml(yaml_str)

        assert isinstance(deserialized, pa.TensorDictSchema)
        assert len(deserialized.keys) == len(sample_schema.keys)

    def test_serialize_deserialize_json(self, sample_schema):
        """Test JSON serialization/deserialization."""
        # Serialize to JSON string
        json_str = pa.to_json(sample_schema)

        # Deserialize from JSON string
        deserialized = pa.from_json(json_str)

        assert isinstance(deserialized, pa.TensorDictSchema)
        assert len(deserialized.keys) == len(sample_schema.keys)

    def test_serialize_with_path_object(self, sample_schema, tmp_path):
        """Test serialization with Path objects."""
        yaml_path = tmp_path / "schema.yaml"

        pa.to_yaml(sample_schema, yaml_path)

        deserialized = pa.from_yaml(yaml_path)

        assert isinstance(deserialized, pa.TensorDictSchema)

    def test_save_load_with_tensor_dict(
        self, sample_schema, sample_tensor_dict, tmp_path
    ):
        """Test saving/loading TensorDict with schema."""
        save_path = tmp_path / "data.pt"

        # Save
        pa.save(sample_schema, sample_tensor_dict, save_path)

        # Load and validate
        loaded = pa.load(save_path)

        assert isinstance(loaded, TensorDict)

    def test_save_load_with_custom_batch_size(self, tmp_path):
        """Test save/load with custom batch sizes."""
        schema = pa.TensorDictSchema(
            keys={
                "data": pa.Tensor(dtype=torch.float32, shape=(None, 100)),
            },
            batch_size=(64,),
        )

        td = TensorDict(
            {
                "data": torch.randn(64, 100),
            },
            batch_size=[64],
        )

        save_path = tmp_path / "custom.pt"
        pa.save(schema, td, save_path)

        loaded = pa.load(save_path)
        # batch_size is returned as torch.Size
        assert tuple(loaded.batch_size) == (64,)


class TestTensorClassSerialization:
    """Test tensorclass serialization functionality."""

    def test_infer_schema_with_tensorclass(self):
        """Test schema inference for tensorclass objects."""

        @tensorclass
        class RLData:
            observation: torch.Tensor
            reward: torch.Tensor

        tc = RLData(
            observation=torch.randn(16, 8),
            reward=torch.randn(16),
            batch_size=[16],
        )

        schema = pa.infer_schema(tc)

        assert isinstance(schema, pa.TensorDictSchema)
        assert len(schema.keys) == 2

    def test_infer_schema_tensorclass_keys(self):
        """Test that tensorclass keys are correctly inferred."""

        @tensorclass
        class Data:
            x: torch.Tensor
            y: torch.Tensor

        tc = Data(x=torch.randn(8, 4), y=torch.randn(8), batch_size=[8])

        schema = pa.infer_schema(tc)

        # Check that both keys are present (tensorclass attributes)
        assert "x" in schema.keys
        assert "y" in schema.keys

    def test_tensorclass_batch_size_inference(self):
        """Test that tensorclass batch_size is correctly inferred."""

        @tensorclass
        class Data:
            obs: torch.Tensor

        tc = Data(obs=torch.randn(24, 10), batch_size=[24])

        schema = pa.infer_schema(tc)
        assert tuple(schema.batch_size) == (24,)
