"""Utility functions for importing optional dependencies."""

from collections.abc import Callable
from functools import wraps
from typing import TypeVar, cast

F = TypeVar("F", bound=Callable)


def strategy_import_error(fn: F) -> F:
    """Decorator to generate input error if dependency is missing."""

    @wraps(fn)
    def _wrapper(*args, **kwargs):
        try:
            import hypothesis
        except ImportError as exc:
            raise ImportError(
                'Strategies for generating data requires "hypothesis" to be \n'
                "installed. You can install pandera together with the strategies \n"
                "dependencies with:\n"
                "pip install pandera[strategies]"
            ) from exc

        return fn(*args, **kwargs)

    return cast(F, _wrapper)
