"""TensorDict container specification."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from pandera.api.base.schema import BaseSchema
from pandera.api.tensordict.components import Tensor


class TensorDictSchema(BaseSchema):
    """Validate TensorDict and tensorclass objects.
    
    Analogous to :class:`~pandera.api.pandas.DataFrameSchema` for DataFrames
    and :class:`~pandera.api.xarray.DatasetSchema` for xarray Datasets.
    
    The schema accepts a ``keys`` parameter (instead of ``columns``) which is
    more semantically appropriate for TensorDict's dictionary-like structure.
    """

    def __init__(
        self,
        keys: Optional[dict[str, Tensor]] = None,
        batch_size: tuple[int | None, ...] | None = None,
        dtype: Any = None,
        checks=None,
        coerce: bool = False,
        name: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Create TensorDict schema.

        :param keys: Dictionary mapping key names to :class:`Tensor` components.
            If a list of strings is provided, creates default Tensor components
            for each key.
        :param batch_size: Expected batch dimensions. Use ``None`` for flexible
            dimensions. For example, ``(32,)`` requires exactly 32 samples,
            while ``(None,)`` allows any batch size.
        :param dtype: Override dtype for all tensors (optional).
        :param checks: Schema-wide checks applied to the entire TensorDict.
        :param coerce: Enable automatic dtype coercion during validation.
        :param name: Name of the schema.
        :param title: Human-readable title for the schema.
        :param description: Detailed description of the schema.
        :param metadata: Additional metadata associated with the schema.
        """
        if keys is None:
            keys = {}
        elif isinstance(keys, list):
            keys = {k: Tensor() for k in keys}
        
        super().__init__(
            dtype=dtype,
            checks=checks,
            coerce=coerce,
            name=name,
            title=title,
            description=description,
            metadata=metadata,
        )
        
        self.keys: dict[str, Tensor] = keys
        self.batch_size = batch_size
    
    @staticmethod
    def register_default_backends(check_obj_cls: type) -> None:
        """Register default backends at validation time."""
        from pandera.backends.tensordict.register import (
            register_tensordict_backends,
        )
        register_tensordict_backends()
    
    @classmethod
    def get_backend(cls, check_obj: Any | None = None) -> BaseSchemaBackend:
        """Get the backend for TensorDict validation.
        
        Overrides base class to handle tensorclass objects which have unique
        types at runtime. We use TensorDict's backend since tensorclasses share
        the same interface and both inherit from object in their MRO.
        """
        if check_obj is not None:
            try:
                # Check if it's a valid TensorDict/tensorclass object
                from pandera.backends.tensordict.base import _is_tensordict
                
                if _is_tensordict(check_obj):
                    check_type = type(check_obj)
                    cls.register_default_backends(check_type)
                    
                    # Try to get backend for this exact type first
                    classes = __import__('inspect').getmro(check_type)
                    for _class in classes:
                        try:
                            return cls.BACKEND_REGISTRY[(cls, _class)]()
                        except KeyError:
                            pass
                    
                    # For tensorclass instances (which don't have a specific backend),
                    # fall back to TensorDict's backend
                    from tensordict import TensorDict as TD
                    if check_type != TD and hasattr(check_obj, '_is_tensorclass'):
                        try:
                            return cls.BACKEND_REGISTRY[(cls, TD)]()
                        except KeyError:
                            pass
            except ImportError:
                pass
        
        # Default fallback to base class implementation
        return super().get_backend(check_obj)
    
    def validate(
        self,
        check_obj: Any,
        head: int | None = None,
        tail: int | None = None,
        sample: int | None = None,
        random_state: int | None = None,
        lazy: bool = False,
        inplace: bool = False,
    ) -> Any:
        """Validate a TensorDict against the schema.

        :param check_obj: TensorDict or tensorclass to validate.
        :param head: Number of leading samples to validate (not used for TensorDict).
        :param tail: Number of trailing samples to validate (not used for TensorDict).
        :param sample: Number of random samples to validate (not used for TensorDict).
        :param random_state: Random seed for sampling (not used for TensorDict).
        :param lazy: If ``True``, collect all errors instead of fail-fast.
        :param inplace: Whether to modify object in place (not supported for TensorDict).
        :returns: Validated TensorDict or tensorclass.
        :raises SchemaError: If validation fails.
        """
        return self.get_backend(check_obj).validate(
            check_obj=check_obj,
            schema=self,
            head=head,
            tail=tail,
            sample=sample,
            random_state=random_state,
            lazy=lazy,
            inplace=inplace,
        )

    def __repr__(self) -> str:
        """Return string representation of the schema."""
        keys_str = ", ".join(repr(k) for k in self.keys.keys())
        return f"TensorDictSchema(keys={{{keys_str}}}, batch_size={self.batch_size})"
