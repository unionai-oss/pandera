"""Register TensorDict backends."""

from functools import lru_cache
from typing import TYPE_CING, Any

if TYPE_CING:
    from tensordict import TensorDict, tensorclass


@lru_cache
def register_tensordict_backends(
    check_cls_fqn: str | None = None,
):
    """Register TensorDict backends.

    This function is called at schema initialization in the _register_*_backends
    method.
    """
    try:
        from tensordict import TensorDict, tensorclass
    except ImportError:
        return

    from pandera.api.tensordict.container import TensorDictSchema
    from pandera.api.tensordict.components import Tensor
    from pandera.api.checks import Check
    from pandera.backends.tensordict.base import TensorDictSchemaBackend
    from pandera.backends.tensordict.checks import TensorDictCheckBackend

    TensorDictSchema.register_backend(
        TensorDict, TensorDictSchemaBackend
    )
    TensorDictSchema.register_backend(
        tensorclass, TensorDictSchemaBackend
    )
    Tensor.register_backend(
        TensorDict, TensorDictSchemaBackend
    )
    Tensor.register_backend(
        tensorclass, TensorDictSchemaBackend
    )
    Check.register_backend(
        TensorDict, TensorDictCheckBackend
    )
    Check.register_backend(
        tensorclass, TensorDictCheckBackend
    )
