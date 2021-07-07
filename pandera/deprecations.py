"""Utility functions for deprecating features."""

import inspect
import warnings
from functools import wraps

from pandera.errors import SchemaInitError


def deprecate_pandas_dtype(fn):
    """
    __init__ decorator for raising SchemaInitError or warnings based on
    the dtype and pandas_dtype input.
    """

    @wraps(fn)
    def wrapper(*args, **kwargs):
        """__init__ method wrapper for raising deprecation warning."""
        sig = inspect.signature(fn)
        bound_args = sig.bind(*args, **kwargs)
        dtype = bound_args.arguments.get("dtype", None)
        pandas_dtype = bound_args.arguments.get("pandas_dtype", None)

        msg = (
            "`pandas_dtype` is deprecated and will be removed as an "
            "option in pandera v0.9.0, use `dtype` instead."
        )

        if dtype is not None and pandas_dtype is not None:
            raise SchemaInitError(
                f"`dtype` and `pandas_dtype` cannot both be specified. {msg}"
            )
        if pandas_dtype is not None:
            warnings.warn(msg, DeprecationWarning)

        return fn(*args, **kwargs)

    return wrapper
