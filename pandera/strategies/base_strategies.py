"""Base module for `hypothesis`-based strategies for data synthesis."""

from typing import Callable, Dict, Generic, Tuple, Type, TypeVar


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
