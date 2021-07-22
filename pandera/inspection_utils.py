"""Decorators for integrating pandera into existing data pipelines."""
from inspect import ismethod
from typing import Callable


def _is_like_classmethod(fn: Callable) -> bool:
    """A regular method defined on a metaclass behaves the same way as
    a method decorated with @classmethod defined on a regular class.

    This function covers both use cases.
    """
    is_method = ismethod(fn)
    return is_method and isinstance(fn.__self__, type)  # type: ignore[attr-defined]


def is_decorated_classmethod(fn: Callable) -> bool:
    """Check if fn is a classmethod declared with the @classmethod decorator.

    Adapted from:
    https://stackoverflow.com/questions/19227724/check-if-a-function-uses-classmethod
    """
    if not _is_like_classmethod(fn):
        return False
    bound_to = fn.__self__  # type: ignore[attr-defined]
    assert isinstance(bound_to, type)
    name = fn.__name__
    for cls in bound_to.__mro__:
        descriptor = vars(cls).get(name)
        if descriptor is not None:
            return isinstance(descriptor, classmethod)
    return False


def is_classmethod_from_meta(fn: Callable) -> bool:
    """Check if fn is a regular method defined on a metaclass
    (which behaves like an @classmethod method defined on a regular class)."""
    return not is_decorated_classmethod(fn) and _is_like_classmethod(fn)
