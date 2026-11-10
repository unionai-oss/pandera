"""Hypothesis-based data synthesis (import backend modules explicitly).

Example: ``import pandera.strategies.pandas_strategies`` or
``import pandera.strategies.xarray_strategies``.
"""

from typing import Any

_TENSORDICT_STRATEGIES = ("tensordict_strategy", "tensorclass_strategy")


def __getattr__(name: str) -> Any:
    # Lazily expose tensordict strategies so that importing this package
    # doesn't require torch/tensordict to be installed.
    if name in _TENSORDICT_STRATEGIES:
        from pandera.strategies import tensordict_strategies

        return getattr(tensordict_strategies, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
