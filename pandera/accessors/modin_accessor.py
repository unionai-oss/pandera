"""Custom accessor functionality for modin.

Source code adapted from pyspark.pandas implementation:
https://spark.apache.org/docs/3.2.0/api/python/reference/pyspark.pandas/api/pyspark.pandas.extensions.register_dataframe_accessor.html?highlight=register_dataframe_accessor#pyspark.pandas.extensions.register_dataframe_accessor
"""

import warnings

from pandera.accessors.pandas_accessor import (
    PanderaDataFrameAccessor,
    PanderaSeriesAccessor,
)


# pylint: disable=too-few-public-methods
class CachedAccessor:
    """
    Custom property-like object.

    A descriptor for caching accessors:

    :param name: Namespace that accessor's methods, properties, etc will be
        accessed under, e.g. "foo" for a dataframe accessor yields the accessor
        ``df.foo``
    :param cls: Class with the extension methods.

    For accessor, the class's __init__ method assumes that you are registering
    an accessor for one of ``Series``, ``DataFrame``, or ``Index``.
    """

    def __init__(self, name, accessor):
        self._name = name
        self._accessor = accessor

    def __get__(self, obj, cls):
        if obj is None:  # pragma: no cover
            return self._accessor
        accessor_obj = self._accessor(obj)
        object.__setattr__(obj, self._name, accessor_obj)
        return accessor_obj


def _register_accessor(name, cls):
    """
    Register a custom accessor on {class} objects.

    :param name: Name under which the accessor should be registered. A warning
        is issued if this name conflicts with a preexisting attribute.
    :returns: A class decorator callable.
    """

    def decorator(accessor):
        if hasattr(cls, name):
            msg = (
                f"registration of accessor {accessor} under name '{name}' for "
                "type {cls.__name__} is overriding a preexisting attribute "
                "with the same name."
            )

            warnings.warn(
                msg,
                UserWarning,
                stacklevel=2,
            )
        setattr(cls, name, CachedAccessor(name, accessor))
        return accessor

    return decorator


def register_dataframe_accessor(name):
    """
    Register a custom accessor with a DataFrame

    :param name: name used when calling the accessor after its registered
    :returns: a class decorator callable.
    """
    # pylint: disable=import-outside-toplevel
    from modin.pandas import DataFrame

    return _register_accessor(name, DataFrame)


def register_series_accessor(name):
    """
    Register a custom accessor with a Series object

    :param name: name used when calling the accessor after its registered
    :returns: a callable class decorator
    """
    # pylint: disable=import-outside-toplevel
    from modin.pandas import Series

    return _register_accessor(name, Series)


register_dataframe_accessor("pandera")(PanderaDataFrameAccessor)
register_series_accessor("pandera")(PanderaSeriesAccessor)
