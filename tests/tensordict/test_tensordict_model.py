"""Unit tests for TensorDictModel."""

import pytest

try:
    import torch
    from tensordict import TensorDict
    from pandera.tensordict import DataType
except ImportError:
    torch = None
    TensorDict = None
    DataType = None

torch_condition = pytest.mark.skipif(torch is None, reason="torch not installed")


@torch_condition
class TestTensorDictModel:
    """Tests for TensorDictModel."""

    def test_model_basic(self):
        """Test basic TensorDictModel."""
        from pandera.tensordict import DataType, TensorDictModel

        class BasicModel(TensorDictModel):
            observation: DataType
            action: DataType

        schema = BasicModel.to_schema()
        assert "observation" in schema.columns
        assert "action" in schema.columns

    def test_model_with_field(self):
        """Test TensorDictModel with Field."""
        from pandera.tensordict import DataType, Field, TensorDictModel

        class ModelWithField(TensorDictModel):
            observation: DataType = Field(dtype=torch.float32, shape=(None, 10))
            action: DataType = Field(dtype=torch.float32, shape=(None, 5))

        schema = ModelWithField.to_schema()
        assert "observation" in schema.columns
        assert "action" in schema.columns
        assert str(schema.columns["observation"].dtype) == "torch.float32"

    def test_model_with_config(self):
        """Test TensorDictModel with Config."""
        from pandera.tensordict import DataType, TensorDictModel

        class ModelWithConfig(TensorDictModel):
            observation: DataType
            action: DataType

            class Config:
                batch_size = (32,)

        schema = ModelWithConfig.to_schema()
        assert schema.batch_size == (32,)

    def test_model_with_checks(self):
        """Test TensorDictModel with check parameters."""
        from pandera.tensordict import DataType, Field, TensorDictModel

        class ModelWithChecks(TensorDictModel):
            values: DataType = Field(
                dtype=torch.float32,
                shape=(None,),
                gt=0.0,
                lt=100.0,
            )

        schema = ModelWithChecks.to_schema()
        assert "values" in schema.columns
        assert len(schema.columns["values"].checks) == 2

    def test_model_validate(self):
        """Test TensorDictModel validate method."""
        from pandera.tensordict import DataType, Field, TensorDictModel

        class ModelForValidation(TensorDictModel):
            observation: DataType = Field(dtype=torch.float32, shape=(32, 10))
            action: DataType = Field(dtype=torch.float32, shape=(32, 5))

            class Config:
                batch_size = (32,)

        td = TensorDict(
            {"observation": torch.randn(32, 10), "action": torch.randn(32, 5)},
            batch_size=[32],
        )

        result = ModelForValidation.validate(td)
        assert isinstance(result, TensorDict)

    def test_model_validate_invalid(self):
        """Test TensorDictModel validate raises on invalid data."""
        from pandera import errors
        from pandera.tensordict import DataType, Field, TensorDictModel

        class ModelForValidation(TensorDictModel):
            values: DataType = Field(dtype=torch.float32, shape=(10,))

            class Config:
                batch_size = (10,)

        td = TensorDict(
            {"values": torch.randn(10, 10)},
            batch_size=[10],
        )

        with pytest.raises(errors.SchemaErrors):
            ModelForValidation.validate(td, lazy=True)

    def test_model_optional_field(self):
        """Test TensorDictModel with optional field."""
        from pandera.tensordict import DataType, TensorDictModel

        class ModelWithOptional(TensorDictModel):
            observation: DataType
            action: DataType | None

        schema = ModelWithOptional.to_schema()
        assert "observation" in schema.columns
        assert "action" in schema.columns

    def test_model_field_with_no_nan(self):
        """Test TensorDictModel with no_nan check."""
        from pandera.tensordict import DataType, Field, TensorDictModel

        class ModelWithNoNan(TensorDictModel):
            values: DataType = Field(
                dtype=torch.float32,
                shape=(None,),
                gt=0.0,
            )

        schema = ModelWithNoNan.to_schema()
        assert "values" in schema.columns
        assert len(schema.columns["values"].checks) == 1

    def test_model_field_with_isin(self):
        """Test TensorDictModel with isin check."""
        from pandera.tensordict import DataType, Field, TensorDictModel

        class ModelWithIsin(TensorDictModel):
            actions: DataType = Field(
                dtype=torch.int64,
                shape=(None,),
                isin=[0, 1, 2, 3],
            )

        schema = ModelWithIsin.to_schema()
        assert "actions" in schema.columns
        assert len(schema.columns["actions"].checks) == 1


@torch_condition
class TestTensorDictModelEdgeCases:
    """Edge case tests for TensorDictModel."""

    def test_model_missing_annotation(self):
        """Test model raises error with missing annotation."""
        from pandera import errors
        from pandera.tensordict import TensorDictModel

        class ModelWithMissingAnnotation(TensorDictModel):
            observation: None  # type: ignore

        with pytest.raises(errors.SchemaInitError):
            ModelWithMissingAnnotation.to_schema()

    def test_model_empty(self):
        """Test model with no fields."""
        from pandera.tensordict import TensorDictModel

        class EmptyModel(TensorDictModel):
            pass

        schema = EmptyModel.to_schema()
        assert schema.columns == {}

    def test_model_cannot_instantiate_directly(self):
        """Test that TensorDictModel cannot be instantiated directly."""
        from pandera.tensordict import TensorDictModel

        with pytest.raises(NotImplementedError):
            TensorDictModel()


@torch_condition
class TestTensorDictModelErrorCases:
    """Comprehensive error case tests for TensorDictModel."""

    def test_model_validate_wrong_batch_size(self):
        """Test model validate fails with wrong batch size."""
        from pandera import errors
        from pandera.tensordict import DataType, Field, TensorDictModel

        class ModelForValidation(TensorDictModel):
            observation: DataType = Field(dtype=torch.float32, shape=(10, 10))

            class Config:
                batch_size = (10,)

        td = TensorDict(
            {"observation": torch.randn(16, 10)},
            batch_size=[16],
        )

        with pytest.raises(errors.SchemaErrors):
            ModelForValidation.validate(td, lazy=True)

    def test_model_validate_wrong_dtype(self):
        """Test model validate fails with wrong dtype."""
        from pandera import errors
        from pandera.tensordict import DataType, Field, TensorDictModel

        class ModelForValidation(TensorDictModel):
            observation: DataType = Field(dtype=torch.float32, shape=(10, 10))

            class Config:
                batch_size = (10,)

        td = TensorDict(
            {"observation": torch.randn(10, 10).to(torch.int32)},
            batch_size=[10],
        )

        with pytest.raises(errors.SchemaErrors):
            ModelForValidation.validate(td, lazy=True)

    def test_model_validate_missing_key(self):
        """Test model validate fails with missing key."""
        from pandera import errors
        from pandera.tensordict import DataType, Field, TensorDictModel

        class ModelForValidation(TensorDictModel):
            observation: DataType = Field(dtype=torch.float32, shape=(10, 10))
            action: DataType = Field(dtype=torch.float32, shape=(10, 5))

            class Config:
                batch_size = (10,)

        td = TensorDict(
            {"observation": torch.randn(10, 10)},
            batch_size=[10],
        )

        with pytest.raises(errors.SchemaErrors):
            ModelForValidation.validate(td, lazy=True)

    def test_model_validate_wrong_shape(self):
        """Test model validate fails with wrong shape."""
        from pandera import errors
        from pandera.tensordict import DataType, Field, TensorDictModel

        class ModelForValidation(TensorDictModel):
            observation: DataType = Field(dtype=torch.float32, shape=(10, 10))

            class Config:
                batch_size = (10,)

        td = TensorDict(
            {"observation": torch.randn(10, 20)},
            batch_size=[10],
        )

        with pytest.raises(errors.SchemaErrors):
            ModelForValidation.validate(td, lazy=True)

    def test_model_validate_missing_key(self):
        """Test model validate fails with missing key."""
        from pandera import errors
        from pandera.tensordict import DataType, Field, TensorDictModel

        class ModelForValidation(TensorDictModel):
            observation: DataType = Field(dtype=torch.float32, shape=(32, 10))
            action: DataType = Field(dtype=torch.float32, shape=(32, 5))

            class Config:
                batch_size = (32,)

        td = TensorDict(
            {"observation": torch.randn(32, 10)},
            batch_size=[32],
        )

        with pytest.raises(errors.SchemaErrors):
            ModelForValidation.validate(td, lazy=True)

    def test_model_validate_wrong_shape(self):
        """Test model validate fails with wrong shape."""
        from pandera import errors
        from pandera.tensordict import DataType, Field, TensorDictModel

        class ModelForValidation(TensorDictModel):
            observation: DataType = Field(dtype=torch.float32, shape=(32, 10))

            class Config:
                batch_size = (32,)

        td = TensorDict(
            {"observation": torch.randn(32, 20)},
            batch_size=[32],
        )

        with pytest.raises(errors.SchemaErrors):
            ModelForValidation.validate(td, lazy=True)

    def test_model_validate_value_check_failure(self):
        """Test model validate fails with value check failure."""
        from pandera import errors
        from pandera.tensordict import DataType, Field, TensorDictModel

        class ModelForValidation(TensorDictModel):
            values: DataType = Field(
                dtype=torch.float32,
                shape=(10,),
                gt=0.0,
            )

            class Config:
                batch_size = (10,)

        td = TensorDict(
            {"values": torch.tensor([1.0, -2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])},
            batch_size=[10],
        )

        with pytest.raises(errors.SchemaErrors):
            ModelForValidation.validate(td, lazy=True)

    def test_model_config_batch_size(self):
        """Test model config batch_size is used."""
        from pandera.tensordict import DataType, Field, TensorDictModel

        class ModelWithConfig(TensorDictModel):
            observation: DataType = Field(dtype=torch.float32, shape=(32, 10))

            class Config:
                batch_size = (32,)

        schema = ModelWithConfig.to_schema()
        assert schema.batch_size == (32,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
