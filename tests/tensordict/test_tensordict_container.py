"""Unit tests for TensorDict container and components."""

import pytest

try:
    import torch
    from tensordict import TensorDict, tensorclass
except ImportError:
    torch = None
    TensorDict = None
    tensorclass = None

torch_condition = pytest.mark.skipif(torch is None, reason="torch not installed")


@torch_condition
class TestTensorComponent:
    """Tests for Tensor component."""

    def test_tensor_creation(self):
        """Test Tensor component creation."""
        from pandera.tensordict import Tensor

        tensor = Tensor(dtype=torch.float32, shape=(None, 10))
        assert tensor.dtype == torch.float32
        assert tensor.shape == (None, 10)

    def test_tensor_with_checks(self):
        """Test Tensor component with checks."""
        from pandera import Check
        from pandera.tensordict import Tensor

        tensor = Tensor(
            dtype=torch.float32,
            shape=(None, 10),
            checks=Check.greater_than(0.0),
        )
        assert tensor.dtype == torch.float32
        assert tensor.shape == (None, 10)
        assert len(tensor.checks) == 1

    def test_tensor_repr(self):
        """Test Tensor repr."""
        from pandera.tensordict import Tensor

        tensor = Tensor(dtype=torch.float32, shape=(None, 10))
        repr_str = repr(tensor)
        assert "Tensor" in repr_str
        assert "float32" in repr_str


@torch_condition
class TestTensorDictSchema:
    """Tests for TensorDictSchema."""

    def test_tensordict_schema_creation(self):
        """Test TensorDictSchema creation."""
        from pandera.tensordict import Tensor, TensorDictSchema

        schema = TensorDictSchema(
            keys={
                "observation": Tensor(dtype=torch.float32, shape=(None, 10)),
                "action": Tensor(dtype=torch.float32, shape=(None, 5)),
            },
            batch_size=(32,),
        )
        assert "observation" in schema.columns
        assert "action" in schema.columns
        assert schema.batch_size == (32,)

    def test_tensordict_schema_from_list(self):
        """Test TensorDictSchema creation from key list."""
        from pandera.tensordict import TensorDictSchema

        schema = TensorDictSchema(keys=["observation", "action"], batch_size=(32,))
        assert "observation" in schema.columns
        assert "action" in schema.columns

    def test_tensordict_schema_repr(self):
        """Test TensorDictSchema repr."""
        from pandera.tensordict import Tensor, TensorDictSchema

        schema = TensorDictSchema(
            keys={
                "observation": Tensor(dtype=torch.float32),
            },
            batch_size=(32,),
        )
        repr_str = repr(schema)
        assert "TensorDictSchema" in repr_str
        assert "observation" in repr_str


@torch_condition
class TestTensorDictValidation:
    """Tests for TensorDict validation."""

    def test_validate_valid_tensordict(self):
        """Test validation of valid TensorDict."""
        from pandera.tensordict import Tensor, TensorDictSchema

        schema = TensorDictSchema(
            keys={
                "observation": Tensor(dtype=torch.float32, shape=(32, 10)),
                "action": Tensor(dtype=torch.float32, shape=(32, 5)),
            },
            batch_size=(32,),
        )

        td = TensorDict(
            {
                "observation": torch.randn(32, 10),
                "action": torch.randn(32, 5),
            },
            batch_size=[32],
        )

        result = schema.validate(td)
        assert isinstance(result, TensorDict)

    def test_validate_invalid_batch_size(self):
        """Test validation fails with wrong batch size."""
        from pandera import errors
        from pandera.tensordict import Tensor, TensorDictSchema

        schema = TensorDictSchema(
            keys={
                "observation": Tensor(dtype=torch.float32, shape=(32, 10)),
            },
            batch_size=(32,),
        )

        td = TensorDict(
            {"observation": torch.randn(16, 10)},
            batch_size=[16],
        )

        with pytest.raises(errors.SchemaErrors):
            schema.validate(td)

    def test_validate_missing_key(self):
        """Test validation fails with missing key."""
        from pandera import errors
        from pandera.tensordict import Tensor, TensorDictSchema

        schema = TensorDictSchema(
            keys={
                "observation": Tensor(dtype=torch.float32),
                "action": Tensor(dtype=torch.float32),
            },
            batch_size=(32,),
        )

        td = TensorDict(
            {"observation": torch.randn(32, 10)},
            batch_size=[32],
        )

        with pytest.raises(errors.SchemaErrors):
            schema.validate(td)

    def test_validate_wrong_dtype(self):
        """Test validation fails with wrong dtype."""
        from pandera import errors
        from pandera.tensordict import Tensor, TensorDictSchema

        schema = TensorDictSchema(
            keys={
                "observation": Tensor(dtype=torch.float32),
            },
            batch_size=(32,),
        )

        td = TensorDict(
            {"observation": torch.randn(32, 10).to(torch.float64)},
            batch_size=[32],
        )

        with pytest.raises(errors.SchemaErrors):
            schema.validate(td)

    def test_validate_wrong_shape(self):
        """Test validation fails with wrong shape."""
        from pandera import errors
        from pandera.tensordict import Tensor, TensorDictSchema

        schema = TensorDictSchema(
            keys={
                "observation": Tensor(dtype=torch.float32, shape=(32, 10)),
            },
            batch_size=(32,),
        )

        td = TensorDict(
            {"observation": torch.randn(32, 20)},
            batch_size=[32],
        )

        with pytest.raises(errors.SchemaErrors):
            schema.validate(td)

    def test_validate_lazy(self):
        """Test lazy validation collects all errors."""
        from pandera import errors
        from pandera.tensordict import Tensor, TensorDictSchema

        schema = TensorDictSchema(
            keys={
                "observation": Tensor(dtype=torch.float32, shape=(32, 10)),
                "action": Tensor(dtype=torch.float32, shape=(32, 5)),
            },
            batch_size=(32,),
        )

        td = TensorDict(
            {"observation": torch.randn(16, 10), "action": torch.randn(32, 5)},
            batch_size=[16],
        )

        with pytest.raises(errors.SchemaErrors) as exc_info:
            schema.validate(td, lazy=True)

        assert len(exc_info.value.schema_errors) > 0


@torch_condition
class TestTensorDictSchemaChecks:
    """Tests for TensorDictSchema with value checks."""

    def test_validate_with_value_check(self):
        """Test validation with value check."""
        from pandera import Check
        from pandera.tensordict import Tensor, TensorDictSchema

        schema = TensorDictSchema(
            keys={
                "values": Tensor(
                    dtype=torch.float32,
                    shape=(None,),
                    checks=Check.greater_than(0.0),
                ),
            },
            batch_size=(10,),
        )

        td = TensorDict(
            {"values": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])},
            batch_size=[10],
        )

        result = schema.validate(td)
        assert isinstance(result, TensorDict)

    def test_validate_value_check_failure(self):
        """Test validation fails with value check failure."""
        from pandera import Check, errors
        from pandera.tensordict import Tensor, TensorDictSchema

        schema = TensorDictSchema(
            keys={
                "values": Tensor(
                    dtype=torch.float32,
                    shape=(None,),
                    checks=Check.greater_than(0.0),
                ),
            },
            batch_size=(10,),
        )

        td = TensorDict(
            {"values": torch.tensor([1.0, -2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])},
            batch_size=[10],
        )

        with pytest.raises(errors.SchemaErrors):
            schema.validate(td)


@torch_condition
class TestTensorDictSchemaBatchSize:
    """Tests for batch size validation."""

    def test_batch_size_with_none_dimension(self):
        """Test batch size with None dimension allows any size."""
        from pandera.tensordict import Tensor, TensorDictSchema

        schema = TensorDictSchema(
            keys={
                "observation": Tensor(dtype=torch.float32, shape=(None, 10)),
            },
            batch_size=(None,),
        )

        td = TensorDict({"observation": torch.randn(5, 10)}, batch_size=[5])
        result = schema.validate(td)
        assert isinstance(result, TensorDict)

        td = TensorDict({"observation": torch.randn(100, 10)}, batch_size=[100])
        result = schema.validate(td)
        assert isinstance(result, TensorDict)

    def test_batch_size_exact_match(self):
        """Test batch size must match exactly when specified."""
        from pandera import errors
        from pandera.tensordict import Tensor, TensorDictSchema

        schema = TensorDictSchema(
            keys={
                "observation": Tensor(dtype=torch.float32, shape=(None, 10)),
            },
            batch_size=(32,),
        )

        td = TensorDict({"observation": torch.randn(64, 10)}, batch_size=[64])

        with pytest.raises(errors.SchemaErrors):
            schema.validate(td)


@torch_condition
class TestShapeValidation:
    """Tests for shape validation."""

    def test_shape_with_none_allows_any(self):
        """Test shape with None allows any size for that dimension."""
        from pandera.tensordict import Tensor, TensorDictSchema

        schema = TensorDictSchema(
            keys={
                "data": Tensor(dtype=torch.float32, shape=(None, None, 3)),
            },
            batch_size=(10,),
        )

        td = TensorDict({"data": torch.randn(10, 32, 3)}, batch_size=[10])
        result = schema.validate(td)
        assert isinstance(result, TensorDict)

    def test_shape_exact_match(self):
        """Test shape must match exactly when specified."""
        from pandera import errors
        from pandera.tensordict import Tensor, TensorDictSchema

        schema = TensorDictSchema(
            keys={
                "data": Tensor(dtype=torch.float32, shape=(32, 10)),
            },
            batch_size=(32,),
        )

        td = TensorDict({"data": torch.randn(32, 20)}, batch_size=[32])

        with pytest.raises(errors.SchemaErrors):
            schema.validate(td)


@torch_condition
class TestTensorClassValidation:
    """Tests for tensorclass validation."""

    def test_validate_tensorclass(self):
        """Test validation of tensorclass object."""
        from pandera.tensordict import Tensor, TensorDictSchema

        @tensorclass
        class TCData:
            observation: torch.Tensor
            action: torch.Tensor

        schema = TensorDictSchema(
            keys={
                "observation": Tensor(dtype=torch.float32, shape=(32, 10)),
                "action": Tensor(dtype=torch.float32, shape=(32, 5)),
            },
            batch_size=(32,),
        )

        tc = TCData(
            observation=torch.randn(32, 10),
            action=torch.randn(32, 5),
            batch_size=[32],
        )

        result = schema.validate(tc)
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
