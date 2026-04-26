"""Generate synthetic TensorDict data from schema definitions.

This module generates :class:`tensordict.TensorDict` and
:class:`tensordict.tensorclass` objects that conform to
:class:`~pandera.api.tensordict.container.TensorDictSchema` specifications.

Built on top of the
`hypothesis <https://hypothesis.readthedocs.io/en/latest/index.html>`_
package.
"""

from __future__ import annotations

from functools import wraps
from typing import Any, TypeVar, cast

import torch

from pandera.strategies.base_strategies import HAS_HYPOTHESIS

if HAS_HYPOTHESIS:
    from hypothesis import strategies as st
    from hypothesis.strategies import SearchStrategy, composite
else:  # pragma: no cover
    from pandera.strategies.base_strategies import SearchStrategy, composite

F = TypeVar("F")


def _strategy_import_error(fn: F) -> F:
    """Decorator to raise ImportError when hypothesis is missing."""

    @wraps(fn)
    def _wrapper(*args, **kwargs):
        if not HAS_HYPOTHESIS:
            raise ImportError(
                'Strategies for generating data requires "hypothesis" '
                "to be installed.\n"
                "pip install pandera[strategies]"
            )
        return fn(*args, **kwargs)

    return cast(F, _wrapper)


@_strategy_import_error
def tensordict_strategy(
    schema,
    size: int | None = None,
) -> SearchStrategy:
    """Create a strategy from a TensorDictSchema.

    :param schema: The TensorDictSchema to use.
    :param size: Default size for None dimensions.
    :returns: hypothesis strategy producing conforming TensorDicts.
    """
    default_size = size or 3

    @composite
    def generate_tensordict(draw):
        from tensordict import TensorDict

        batch_size_tuple = schema.batch_size or (default_size,)

        data = {}

        for key_name, tensor_schema in schema.keys.items():
            shape_list = (
                list(tensor_schema.shape) if tensor_schema.shape else []
            )

            # Resolve dynamic dimensions (None -> default_size)
            resolved_shape = tuple(
                s if s is not None else default_size for s in shape_list
            )

            # Calculate total elements needed from resolved shape
            total_elements = 1
            for dim in resolved_shape:
                total_elements *= dim

            dtype_str = str(tensor_schema.dtype).lower()

            if "bool" in dtype_str:
                arr_data = draw(
                    st.lists(
                        st.booleans(),
                        min_size=total_elements,
                        max_size=total_elements,
                    )
                )
                arr = torch.tensor(arr_data, dtype=torch.bool).reshape(
                    resolved_shape
                )
            elif "float" in dtype_str or "double" in dtype_str:
                arr_data = draw(
                    st.lists(
                        st.floats(allow_nan=False, allow_infinity=False),
                        min_size=total_elements,
                        max_size=total_elements,
                    )
                )
                arr = torch.tensor(arr_data, dtype=torch.float32).reshape(
                    resolved_shape
                )
            else:
                # Integer types
                arr_data = draw(
                    st.lists(
                        st.integers(min_value=0, max_value=1000),
                        min_size=total_elements,
                        max_size=total_elements,
                    )
                )
                arr = torch.tensor(arr_data, dtype=torch.int64).reshape(
                    resolved_shape
                )

            data[key_name] = arr

        return TensorDict(data, batch_size=batch_size_tuple)

    return generate_tensordict()


@_strategy_import_error
def tensorclass_strategy(
    cls,
    schema,
    size: int | None = None,
) -> SearchStrategy:
    """Create a strategy for generating tensorclass instances.

    :param cls: The tensorclass class to generate.
    :param schema: The TensorDictSchema defining the structure.
    :param size: Default size for None dimensions.
    :returns: hypothesis strategy producing conforming tensorclass instances.
    """
    default_size = size or 3

    @composite
    def generate_tensorclass(draw):
        from tensordict import tensorclass as tc

        batch_size_tuple = schema.batch_size or (default_size,)

        data = {}

        for key_name, tensor_schema in schema.keys.items():
            shape_list = (
                list(tensor_schema.shape) if tensor_schema.shape else []
            )

            # Resolve dynamic dimensions
            resolved_shape = tuple(
                s if s is not None else default_size for s in shape_list
            )

            # Calculate total elements needed from resolved shape
            total_elements = 1
            for dim in resolved_shape:
                total_elements *= dim

            dtype_str = str(tensor_schema.dtype).lower()

            if "bool" in dtype_str:
                arr_data = draw(
                    st.lists(
                        st.booleans(),
                        min_size=total_elements,
                        max_size=total_elements,
                    )
                )
                arr = torch.tensor(arr_data, dtype=torch.bool).reshape(
                    resolved_shape
                )
            elif "float" in dtype_str or "double" in dtype_str:
                arr_data = draw(
                    st.lists(
                        st.floats(allow_nan=False, allow_infinity=False),
                        min_size=total_elements,
                        max_size=total_elements,
                    )
                )
                arr = torch.tensor(arr_data, dtype=torch.float32).reshape(
                    resolved_shape
                )
            else:
                # Integer types
                arr_data = draw(
                    st.lists(
                        st.integers(min_value=0, max_value=1000),
                        min_size=total_elements,
                        max_size=total_elements,
                    )
                )
                arr = torch.tensor(arr_data, dtype=torch.int64).reshape(
                    resolved_shape
                )

            data[key_name] = arr

        return cls(**data, batch_size=batch_size_tuple)

    return generate_tensorclass()
