"""TensorDictModel for class-based schema definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Optional

from pandera.api.base.model import BaseModel


class TensorDictModel(BaseModel):
    """Declarative TensorDict schema using class definitions.
    
    Example:
        >>> import torch
        >>> import pandera.tensordict as pa
        
        >>> class MySchema(pa.TensorDictModel):
        ...     observation: pa.DataType = pa.Field(dtype=torch.float32, shape=(None, 10))
        ...     action: pa.DataType = pa.Field(dtype=torch.int64, shape=(None,))
        ...
        ...     class Config:
        ...         batch_size = (32,)
        
        >>> schema = MySchema.to_schema()
    """

    Config: type[object] = object
    __schema__: ClassVar[Any | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Initialize subclass and build schema from class annotations."""
        super().__init_subclass__(**kwargs)
        cls.__schema__ = cls.to_schema()

    @classmethod
    def to_schema(cls: type[TensorDictModel]) -> Any:
        """Create TensorDictSchema from model class.

        :returns: TensorDictSchema with fields converted to Tensor components.
        """
        columns: dict[str, Any] = {}
        batch_size: Optional[tuple[int | None, ...]] = None

        # Get field definitions from class annotations and Field info
        for name, (annotation_info, field_info) in cls.__fields__.items():
            if hasattr(field_info, "shape"):
                # Extract dtype from annotation
                dtype = annotation_info.annotation
                
                # Handle DataType special case - convert to torch.dtype if possible
                if dtype is not None:
                    try:
                        from pandera.engines.tensordict_engine import DataType as _DataType
                        
                        if isinstance(dtype, type) and issubclass(dtype, _DataType):
                            dtype = dtype.type
                    except Exception:
                        pass
                
                # Get shape from Field info
                shape = getattr(field_info, "shape", None)
                
                # Build Tensor component - use Tensor directly since it's not imported yet
                try:
                    from pandera.tensordict import Tensor
                except ImportError:
                    Tensor = None
                
                if Tensor is not None:
                    tensor_kwargs: dict[str, Any] = {
                        "dtype": dtype,
                        "shape": shape,
                        "name": name,
                    }
                    
                    if hasattr(field_info, "checks") and field_info.checks:
                        tensor_kwargs["checks"] = field_info.checks
                    
                    columns[name] = Tensor(**tensor_kwargs)
        
        # Get batch_size from Config
        if hasattr(cls, "Config"):
            batch_size = getattr(cls.Config, "batch_size", None)

        try:
            from pandera.tensordict import TensorDictSchema
            
            return TensorDictSchema(keys=columns, batch_size=batch_size)
        except ImportError:
            return None

    @classmethod
    def validate(
        cls: type[TensorDictModel],
        check_obj: Any,
        head: int | None = None,
        tail: int | None = None,
        sample: int | None = None,
        random_state: int | None = None,
        lazy: bool = False,
        inplace: bool = False,
    ) -> Any:
        """Validate data against model schema.

        :param check_obj: TensorDict or tensorclass to validate.
        :returns: Validated data object.
        """
        if cls.__schema__ is not None:
            return cls.__schema__.validate(
                check_obj=check_obj,
                head=head,
                tail=tail,
                sample=sample,
                random_state=random_state,
                lazy=lazy,
                inplace=inplace,
            )
        raise RuntimeError("Model schema was not initialized")
