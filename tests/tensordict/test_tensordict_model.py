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


@torch_condition
class TestTensorDictModelBasic:
    """Tests for basic TensorDictModel functionality."""

    def test_model_with_field_annotations(self):
        """Test model with field annotations."""
        from pandera.tensordict import Field, TensorDictModel

        class MyModel(TensorDictModel):
            observation: torch.float32 = Field(shape=(None, 10))
            action: torch.int64 = Field(shape=(None,))

            class Config:
                batch_size = (32,)

        schema = MyModel.to_schema()
        assert "observation" in schema.keys
        assert "action" in schema.keys

    def test_model_validation(self):
        """Test model validation with valid data."""
        from pandera.tensordict import Field, TensorDictModel

        class MyModel(TensorDictModel):
            observation: torch.float32 = Field(shape=(None, 10))

            class Config:
                batch_size = (32,)

        batch = TensorDict(
            {"observation": torch.randn(32, 10)},
            batch_size=[32],
        )

        validated = MyModel.validate(batch)
        assert isinstance(validated, TensorDict)

    def test_model_validation_failure(self):
        """Test model validation fails with invalid data."""
        import pandera.errors as errors
        from pandera.tensordict import Field, TensorDictModel

        class MyModel(TensorDictModel):
            observation: torch.float32 = Field(shape=(None, 10))

            class Config:
                batch_size = (32,)

        batch = TensorDict(
            {"observation": torch.randn(16, 10)},
            batch_size=[16],
        )

        with pytest.raises(errors.SchemaError):
            MyModel.validate(batch)


@torch_condition
class TestTensorDictModelWithChecks:
    """Tests for TensorDictModel with value checks."""

    def test_model_with_field_checks(self):
        """Test model with Field-level checks."""
        import pandera.errors as errors
        from pandera import Check
        from pandera.tensordict import Field, TensorDictModel

        class MyModel(TensorDictModel):
            values: torch.float32 = Field(
                shape=(None,),
                ge=0.0,
                le=1.0,
            )

            class Config:
                batch_size = (10,)

        # Valid data
        batch = TensorDict(
            {"values": torch.rand(10)},
            batch_size=[10],
        )
        validated = MyModel.validate(batch)
        assert isinstance(validated, TensorDict)

        # Invalid data - outside range
        batch_invalid = TensorDict(
            {"values": torch.randn(10)},  # Some negative values
            batch_size=[10],
        )

        with pytest.raises(errors.SchemaError):
            MyModel.validate(batch_invalid)


@torch_condition
class TestTensorDictModelInheritance:
    """Tests for TensorDictModel inheritance."""

    def test_model_inheritance(self):
        """Test model can be inherited from."""
        from pandera.tensordict import Field, TensorDictModel

        class BaseModel(TensorDictModel):
            observation: torch.float32 = Field(shape=(None, 10))

            class Config:
                batch_size = (32,)

        class ChildModel(BaseModel):
            action: torch.int64 = Field(shape=(None,))

        # Base schema should still work
        base_schema = BaseModel.to_schema()
        assert "observation" in base_schema.keys

        # Child schema should have both fields
        child_schema = ChildModel.to_schema()
        assert "observation" in child_schema.keys
        assert "action" in child_schema.keys


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
