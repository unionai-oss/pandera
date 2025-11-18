"""Base module for `hypothesis`-based strategies for data synthesis."""

from collections.abc import Callable
from functools import wraps
from typing import Generic, TypeVar, cast

import pandera.backends.base.builtin_checks

F = TypeVar("F", bound=Callable)


try:
    from hypothesis.strategies import SearchStrategy, composite
except ImportError:  # pragma: no cover
    T = TypeVar("T")

    class SearchStrategy(Generic[T]):  # type: ignore
        """placeholder type."""

    def composite(fn):  # type: ignore
        """placeholder composite strategy."""
        return fn

    HAS_HYPOTHESIS = False
else:
    HAS_HYPOTHESIS = True


# This strategy registry maps (check_name, data_type) -> strategy_function
# For example: ("greater_than", pd.DataFrame) -> (<function gt_strategy>)
STRATEGY_DISPATCHER: dict[tuple[str, type], Callable] = {}


def strategy_import_error(fn: F) -> F:
    """Decorator to generate input error if dependency is missing."""

    @wraps(fn)
    def _wrapper(*args, **kwargs):
        if not HAS_HYPOTHESIS:  # pragma: no cover
            raise ImportError(
                'Strategies for generating data requires "hypothesis" to be \n'
                "installed. You can install pandera together with the strategies \n"
                "dependencies with:\n"
                "pip install pandera[strategies]"
            )
        return fn(*args, **kwargs)

    return cast(F, _wrapper)
