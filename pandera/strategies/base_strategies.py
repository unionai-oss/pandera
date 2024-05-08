"""Base module for `hypothesis`-based strategies for data synthesis."""

from functools import wraps
from typing import Callable, Dict, Generic, Tuple, Type, TypeVar, cast

F = TypeVar("F", bound=Callable)


try:
    # pylint: disable=unused-import
    from hypothesis.strategies import SearchStrategy, composite
except ImportError:  # pragma: no cover
    T = TypeVar("T")

    # pylint: disable=too-few-public-methods
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
STRATEGY_DISPATCHER: Dict[Tuple[str, Type], Callable] = {}


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
