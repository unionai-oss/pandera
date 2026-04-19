"""Built-in checks for TensorDict."""

import sys

# Import torch first to get the actual type, not string annotation
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from typing import Any, Iterable

try:
    from pandera.api.extensions import register_builtin_check
    
    if TORCH_AVAILABLE:
        @register_builtin_check(
            aliases=["gt"],
            error="greater_than({min_value})",
        )
        def greater_than(data: torch.Tensor, min_value: Any) -> torch.Tensor:
            """Ensure values are strictly greater than a minimum value.

            :param data: Input tensor data.
            :param min_value: Lower bound to be exceeded.
            """
            return data > min_value

        @register_builtin_check(
            aliases=["ge"],
            error="greater_than_or_equal_to({min_value})",
        )
        def greater_than_or_equal_to(data: torch.Tensor, min_value: Any) -> torch.Tensor:
            """Ensure all values are greater than or equal to a minimum value.

            :param data: Input tensor data.
            :param min_value: Allowed minimum value.
            """
            return data >= min_value

        @register_builtin_check(
            aliases=["lt"],
            error="less_than({max_value})",
        )
        def less_than(data: torch.Tensor, max_value: Any) -> torch.Tensor:
            """Ensure all values are strictly less than a maximum value.

            :param data: Input tensor data.
            :param max_value: Allowed maximum value.
            """
            return data < max_value

        @register_builtin_check(
            aliases=["le"],
            error="less_than_or_equal_to({max_value})",
        )
        def less_than_or_equal_to(data: torch.Tensor, max_value: Any) -> torch.Tensor:
            """Ensure all values are less than or equal to a maximum value.

            :param data: Input tensor data.
            :param max_value: Allowed maximum value.
            """
            return data <= max_value

        @register_builtin_check(
            aliases=["in_range"],
            error="in_range({min_value}, {max_value})",
        )
        def in_range(
            data: torch.Tensor,
            min_value: Any,
            max_value: Any,
            include_min: bool = True,
            include_max: bool = True,
        ) -> torch.Tensor:
            """Ensure all values are within a range.

            :param data: Input tensor data.
            :param min_value: Minimum allowed value.
            :param max_value: Maximum allowed value.
            :param include_min: If True, min_value is included in the range.
            :param include_max: If True, max_value is included in the range.
            """
            if include_min and include_max:
                return (data >= min_value) & (data <= max_value)
            elif include_min:
                return (data >= min_value) & (data < max_value)
            elif include_max:
                return (data > min_value) & (data <= max_value)
            else:
                return (data > min_value) & (data < max_value)

        @register_builtin_check(
            aliases=["isin"],
            error="isin({allowed_values})",
        )
        def isin(data: torch.Tensor, allowed_values: Iterable) -> torch.Tensor:
            """Ensure all values are in a set of allowed values.

            :param data: Input tensor data.
            :param allowed_values: Set of allowed values.
            """
            return torch.isin(data, torch.as_tensor(list(allowed_values)))

except ImportError:
    # extensions not available yet
    def register_builtin_check(fn=None, **kwargs):
        def decorator(f):
            return f
        return decorator if fn is None else decorator(fn)
