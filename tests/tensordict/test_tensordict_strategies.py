"""Tests for TensorDict Hypothesis strategies."""

from __future__ import annotations

import pytest

try:
    from hypothesis import given, settings
    from hypothesis.strategies import composite

    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False

import torch

import pandera.tensordict as pa


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
class TestTensorDictStrategies:
    """Test TensorDict Hypothesis strategies."""

    @pytest.fixture
    def simple_schema(self):
        """Create a simple schema for testing with scalar values (per batch).

        Note: With batch_size=(N,), tensors should have shape (N, ...) to include
        the batch dimension in their shape.
        """
        return pa.TensorDictSchema(
            keys={
                "data": pa.Tensor(dtype=torch.float32, shape=(8,)),
            },
            batch_size=(8,),
        )

    @pytest.fixture
    def complex_schema(self):
        """Create a more complex schema with multiple tensors."""
        return pa.TensorDictSchema(
            keys={
                "obs": pa.Tensor(dtype=torch.float32, shape=(8, 10)),
                "action": pa.Tensor(dtype=torch.int64, shape=(8,)),
                "reward": pa.Tensor(dtype=torch.float32, shape=(8,)),
            },
            batch_size=(8,),
        )

    def test_tensordict_strategy_generates_valid_data(self, simple_schema):
        """Test that strategy generates valid TensorDicts."""
        from pandera.strategies import tensordict_strategy

        @given(tensordict_strategy(simple_schema))
        @settings(max_examples=5)
        def check(td):
            # Validate that generated data conforms to schema
            validated = simple_schema.validate(td)

            assert isinstance(validated, type(td))
            assert "data" in td.keys()

        check()

    def test_strategy_batch_size(self, complex_schema):
        """Test that strategy respects batch_size."""
        from pandera.strategies import tensordict_strategy

        @given(tensordict_strategy(complex_schema))
        @settings(max_examples=5)
        def check(td):
            assert tuple(td.batch_size) == (8,)

        check()

    def test_strategy_dtypes(self, complex_schema):
        """Test that strategy generates correct dtypes."""
        from pandera.strategies import tensordict_strategy

        @given(tensordict_strategy(complex_schema))
        @settings(max_examples=5)
        def check(td):
            assert td["obs"].dtype == torch.float32
            assert td["action"].dtype == torch.int64
            assert td["reward"].dtype == torch.float32

        check()

    def test_strategy_shapes(self, complex_schema):
        """Test that strategy generates correct shapes."""
        from pandera.strategies import tensordict_strategy

        @given(tensordict_strategy(complex_schema))
        @settings(max_examples=5)
        def check(td):
            # Tensors should match schema's full shape (including batch dim)
            assert td["obs"].shape == torch.Size([8, 10])
            assert td["action"].shape == torch.Size([8])

        check()

    def test_tensorclass_strategy(self, complex_schema):
        """Test tensorclass strategy generation."""
        from tensordict import tensorclass

        from pandera.strategies import tensorclass_strategy

        @tensorclass
        class Data:
            obs: torch.Tensor
            action: torch.Tensor
            reward: torch.Tensor

        strategy = tensorclass_strategy(Data, complex_schema)

        @given(strategy)
        @settings(max_examples=5)
        def check(tc):
            assert tuple(tc.batch_size) == (8,)
            assert hasattr(tc, "obs")
            assert hasattr(tc, "action")

        check()
