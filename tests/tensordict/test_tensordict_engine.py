"""Unit tests for TensorDict engine."""

import pytest

try:
    import torch
except ImportError:
    torch = None

torch_condition = pytest.mark.skipif(torch is None, reason="torch not installed")


@torch_condition
class TestTensorDictEngine:
    """Tests for TensorDict engine."""

    def test_engine_import(self):
        """Test that tensordict_engine can be imported."""
        from pandera.engines import tensordict_engine

        assert tensordict_engine is not None

    def test_engine_dtype_from_string(self):
        """Test engine dtype from string."""
        from pandera.engines import tensordict_engine

        dtype = tensordict_engine.Engine.dtype("float32")
        assert dtype is not None

    def test_engine_dtype_from_torch_dtype(self):
        """Test engine dtype from torch.dtype."""
        from pandera.engines import tensordict_engine

        dtype = tensordict_engine.Engine.dtype(torch.float32)
        assert dtype is not None

    def test_engine_dtype_invalid(self):
        """Test engine raises on invalid dtype."""
        from pandera.engines import tensordict_engine

        with pytest.raises(ValueError):
            tensordict_engine.Engine.dtype("invalid_dtype")

    def test_datatype_coerce(self):
        """Test DataType coerce method."""
        from pandera.engines import tensordict_engine

        dtype = tensordict_engine.DataType(torch.float32)
        tensor = torch.randn(10).to(torch.float64)
        coerced = dtype.coerce(tensor)
        assert coerced.dtype == torch.float32

    def test_datatype_coerce_value(self):
        """Test DataType coerce_value method."""
        from pandera.engines import tensordict_engine

        dtype = tensordict_engine.DataType(torch.float32)
        value = dtype.coerce_value(1.0)
        assert isinstance(value, torch.Tensor)

    def test_datatype_str(self):
        """Test DataType str representation."""
        from pandera.engines import tensordict_engine

        dtype = tensordict_engine.DataType(torch.float32)
        assert "float32" in str(dtype)

    def test_datatype_repr(self):
        """Test DataType repr."""
        from pandera.engines import tensordict_engine

        dtype = tensordict_engine.DataType(torch.float32)
        assert "DataType" in repr(dtype)

    def test_engine_multiple_dtypes(self):
        """Test engine with multiple dtypes."""
        from pandera.engines import tensordict_engine

        dtypes = [
            "float32",
            "float64",
            "int32",
            "int64",
            "int16",
            "int8",
            "uint8",
            "bool",
        ]

        for dtype_str in dtypes:
            dtype = tensordict_engine.Engine.dtype(dtype_str)
            assert dtype is not None


@torch_condition
class TestTensorDictEngineNotInstalled:
    """Tests when torch is not installed."""

    def test_no_torch_placeholder(self):
        """Test that engine has placeholder when torch not installed."""
        import pandera.engines.tensordict_engine as engine

        if torch is None:
            assert engine.DataType is None
            assert engine.Engine is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
