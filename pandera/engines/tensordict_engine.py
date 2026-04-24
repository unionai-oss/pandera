"""TensorDict engine and data types."""

import dataclasses
from typing import Any, Union

from pandera import dtypes
from pandera.dtypes import immutable
from pandera.engines import engine

try:
    import torch
except ImportError:
    torch = None


if torch is not None:

    @immutable(init=True)
    class DataType(dtypes.DataType):
        """DataType for boxing PyTorch tensor dtypes."""

        type: torch.dtype = dataclasses.field(
            default=torch.float32, repr=False, init=False
        )

        def __init__(self, dtype: Any):
            super().__init__()
            if isinstance(dtype, torch.dtype):
                actual_dtype = dtype
            elif isinstance(dtype, str):
                dtype_str = dtype.replace("torch.", "")
                actual_dtype = getattr(torch, dtype_str, None)
                if actual_dtype is None:
                    raise ValueError(f"Unknown torch dtype: {dtype}")
            else:
                actual_dtype = torch.dtype(dtype)
            object.__setattr__(self, "type", actual_dtype)

        def __post_init__(self):
            if isinstance(self.type, torch.dtype):
                return
            actual_dtype = getattr(torch, str(self.type), None)
            if actual_dtype is not None:
                object.__setattr__(self, "type", actual_dtype)

        def coerce(self, data_container: torch.Tensor) -> torch.Tensor:
            """Coerce tensor to the specified dtype."""
            return data_container.type(self.type)

        def coerce_value(self, value: Any) -> Any:
            """Coerce a value to the particular type."""
            return torch.tensor(value, dtype=self.type)

        def try_coerce(self, data_container: torch.Tensor) -> torch.Tensor:
            try:
                return self.coerce(data_container)
            except Exception as exc:
                from pandera import errors

                raise errors.ParserError(
                    f"Could not coerce tensor to type {self.type}",
                    failure_cases=None,
                ) from exc

        def __str__(self) -> str:
            return str(self.type)

        def __repr__(self) -> str:
            return f"DataType({self})"

    class Engine(metaclass=engine.Engine, base_pandera_dtypes=DataType):
        """PyTorch TensorDict data type engine."""

        @classmethod
        def dtype(cls, data_type: Any) -> dtypes.DataType:
            """Convert input into a PyTorch-compatible Pandera DataType."""
            if isinstance(data_type, type) and issubclass(data_type, DataType):
                return DataType(str(data_type.type))
            try:
                return engine.Engine.dtype(cls, data_type)
            except (TypeError, ValueError):
                try:
                    if isinstance(data_type, torch.dtype):
                        return DataType(str(data_type))
                    elif isinstance(data_type, str):
                        return DataType(data_type)
                    else:
                        return DataType(data_type)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"Data type '{data_type}' not understood for TensorDict. "
                        f"Expected a torch.dtype or string like 'float32'."
                    ) from None

        @classmethod
        def register_dtype(cls, pandera_dtype_cls: type[DataType]):
            """Register a Pandera DataType for PyTorch dtypes."""
            cls._check_source_dtype(pandera_dtype_cls)
            equivalents = {}
            strict_equivalents: dict[Any, DataType] = {}
            for source_dtype in (pandera_dtype_cls.type,):
                if source_dtype is not None:
                    equivalents[source_dtype] = pandera_dtype_cls  # type: ignore[assignment]

            if equivalents:
                registry = cls._registry[cls]
                registry.equivalents.update(equivalents)
                registry.strict_equivalents.update(strict_equivalents)

    def _register_torch_dtypes():
        """Register all torch dtypes."""
        torch_dtype_names = [
            "float32",
            "float64",
            "int32",
            "int64",
            "int32",
            "int16",
            "int8",
            "uint8",
            "bool",
            "bfloat16",
            "complex64",
            "complex128",
            "float16",
            "quint8",
            "qint8",
            "qint32",
        ]

        for dtype_name in torch_dtype_names:
            try:
                torch_dtype = getattr(torch, dtype_name, None)
                if torch_dtype is not None:
                    Engine.register_dtype(DataType(dtype_name))
            except (AttributeError, ValueError):
                pass

    _register_torch_dtypes()

else:

    class _DataTypePlaceholder:
        pass

    class _EnginePlaceholder:
        pass

    DataType = None  # type: ignore[misc, assignment]
    Engine = None  # type: ignore[misc, assignment]
