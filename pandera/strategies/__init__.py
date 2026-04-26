"""Hypothesis-based data synthesis (import backend modules explicitly).

Example: ``import pandera.strategies.pandas_strategies`` or
``import pandera.strategies.xarray_strategies``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tensordict_strategies import tensordict_strategy, tensorclass_strategy

__all__ = [
    "pandas_strategies",
    "xarray_strategies", 
    "tensordict_strategies",
]

# Import backend-specific strategy modules
try:
    from . import pandas_strategies
except ImportError:
    pass

try:
    from . import xarray_strategies
except ImportError:
    pass

try:
    from . import tensordict_strategies as tensordict_strategies_module
    
    # Re-export functions at module level for convenience
    tensordict_strategy = tensordict_strategies_module.tensordict_strategy
    tensorclass_strategy = tensordict_strategies_module.tensorclass_strategy
    __all__.extend(["tensordict_strategy", "tensorclass_strategy"])
except ImportError:
    pass
