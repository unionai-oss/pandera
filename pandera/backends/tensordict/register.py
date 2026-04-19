"""Register TensorDict backends."""

from functools import lru_cache


@lru_cache
def register_tensordict_backends() -> None:
    """Register TensorDict backends.
    
    This function is called at schema initialization in the _register_*_backends
    method to register validation backends for TensorDict and tensorclass objects.
    """
    from pandera.api.checks import Check
    from pandera.api.tensordict.container import TensorDictSchema
    from pandera.backends.tensordict.base import TensorDictSchemaBackend, _is_tensordict
    from pandera.backends.tensordict.checks import TensorDictCheckBackend
    
    # Import builtin checks to register check functions
    from pandera.backends.tensordict import builtin_checks
    
    try:
        import torch
        from tensordict import TensorDict
        
        # Register schema backend for TensorDict (has a fixed type)
        TensorDictSchema.register_backend(TensorDict, TensorDictSchemaBackend)
        
        # For tensorclass, we handle it in get_backend() since each decorated
        # class creates a unique type. We use TensorDict's backend since they
        # share the same interface.
        
        # Register check backend for tensors
        Check.register_backend(torch.Tensor, TensorDictCheckBackend)
        
    except ImportError:
        pass
