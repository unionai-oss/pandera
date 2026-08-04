"""Generate synthetic TensorDict data from schema definitions.

This module generates :class:`tensordict.TensorDict` and
:class:`tensordict.tensorclass` objects that conform to
:class:`~pandera.api.tensordict.container.TensorDictSchema` specifications.

Built on top of the
`hypothesis <https://hypothesis.readthedocs.io/en/latest/index.html>`_
package.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar, cast

import numpy as np
import torch

from pandera.strategies.base_strategies import HAS_HYPOTHESIS

if HAS_HYPOTHESIS:
    from hypothesis import strategies as st
    from hypothesis.extra import numpy as npst
    from hypothesis.strategies import SearchStrategy, composite
else:  # pragma: no cover
    from pandera.strategies.base_strategies import SearchStrategy, composite

F = TypeVar("F", bound=Callable[..., Any])


def _strategy_import_error(fn: F) -> F:
    """Decorator to raise ImportError when hypothesis is missing."""

    @wraps(fn)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        if not HAS_HYPOTHESIS:
            raise ImportError(
                'Strategies for generating data requires "hypothesis" '
                "to be installed.\n"
                "pip install pandera[strategies]"
            )
        return fn(*args, **kwargs)

    return cast(F, _wrapper)


def _resolve_batch_size(schema, default_size: int) -> tuple[int, ...]:
    """Resolve a schema batch_size into concrete dimensions."""
    if not schema.batch_size:
        return (default_size,)
    return tuple(
        s if s is not None else default_size for s in schema.batch_size
    )


def _check_bounds(checks) -> dict[str, Any]:
    """Extract element-value constraints from built-in checks."""
    bounds: dict[str, Any] = {}
    for check in checks or []:
        stats = check.statistics or {}
        if check.name == "isin":
            bounds["allowed_values"] = list(stats["allowed_values"])
        elif check.name in ("greater_than", "greater_than_or_equal_to"):
            bounds["min_value"] = stats["min_value"]
            bounds["exclude_min"] = check.name == "greater_than"
        elif check.name in ("less_than", "less_than_or_equal_to"):
            bounds["max_value"] = stats["max_value"]
            bounds["exclude_max"] = check.name == "less_than"
        elif check.name == "in_range":
            bounds["min_value"] = stats["min_value"]
            bounds["max_value"] = stats["max_value"]
            bounds["exclude_min"] = not stats.get("include_min", True)
            bounds["exclude_max"] = not stats.get("include_max", True)
    return bounds


def _element_strategy(torch_dtype: torch.dtype, checks) -> SearchStrategy:
    """Build a strategy for single elements honoring built-in checks."""
    bounds = _check_bounds(checks)
    if "allowed_values" in bounds:
        return st.sampled_from(bounds["allowed_values"])

    if torch_dtype == torch.bool:
        return st.booleans()

    if torch_dtype.is_floating_point:
        width = 32 if torch.finfo(torch_dtype).bits <= 32 else 64
        return st.floats(
            min_value=bounds.get("min_value"),
            max_value=bounds.get("max_value"),
            exclude_min=bounds.get("exclude_min", False),
            exclude_max=bounds.get("exclude_max", False),
            allow_nan=False,
            allow_infinity=False,
            width=width,
        )

    if torch_dtype.is_complex:
        return st.complex_numbers(allow_nan=False, allow_infinity=False)

    # Integer types: apply bounds, defaulting to a small non-negative range.
    min_value = bounds.get("min_value", 0)
    max_value = bounds.get("max_value", 1000)
    if bounds.get("exclude_min"):
        min_value += 1
    if bounds.get("exclude_max"):
        max_value -= 1
    return st.integers(min_value=min_value, max_value=max_value)


def _numpy_dtype(torch_dtype: torch.dtype) -> np.dtype:
    """Numpy dtype used to generate values for a torch dtype.

    Falls back to float32 for torch-only floating dtypes (e.g. bfloat16,
    float8) which have no numpy equivalent; the generated array is cast to
    the target dtype afterwards.
    """
    try:
        return torch.empty(0, dtype=torch_dtype).numpy().dtype
    except TypeError:
        return np.dtype("float32")


def _draw_tensor(
    draw,
    tensor_schema,
    default_size: int,
    batch_size: tuple[int, ...] = (),
) -> torch.Tensor:
    """Draw a tensor conforming to a Tensor component's dtype, shape, and
    built-in value checks."""
    shape_list = list(tensor_schema.shape) if tensor_schema.shape else []

    # Resolve dynamic dimensions: leading None dims align with the
    # TensorDict batch dimensions, remaining ones use default_size.
    resolved_shape = tuple(
        (batch_size[i] if i < len(batch_size) else default_size)
        if s is None
        else s
        for i, s in enumerate(shape_list)
    )

    torch_dtype = (
        tensor_schema.dtype.type
        if tensor_schema.dtype is not None
        else torch.float32
    )

    arr = draw(
        npst.arrays(
            dtype=_numpy_dtype(torch_dtype),
            shape=resolved_shape,
            elements=_element_strategy(torch_dtype, tensor_schema.checks),
        )
    )
    return torch.from_numpy(np.ascontiguousarray(arr)).to(torch_dtype)


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

        batch_size = _resolve_batch_size(schema, default_size)
        data = {
            key_name: _draw_tensor(
                draw, tensor_schema, default_size, batch_size
            )
            for key_name, tensor_schema in schema.keys.items()
        }
        return TensorDict(data, batch_size=batch_size)

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
        batch_size = _resolve_batch_size(schema, default_size)
        data = {
            key_name: _draw_tensor(
                draw, tensor_schema, default_size, batch_size
            )
            for key_name, tensor_schema in schema.keys.items()
        }
        return cls(**data, batch_size=batch_size)

    return generate_tensorclass()
