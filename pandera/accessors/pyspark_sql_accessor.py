"""Custom accessor functionality for modin.

Source code adapted from pyspark.pandas implementation:
https://spark.apache.org/docs/3.2.0/api/python/reference/pyspark.pandas/api/pyspark.pandas.extensions.register_dataframe_accessor.html?highlight=register_dataframe_accessor#pyspark.pandas.extensions.register_dataframe_accessor
"""

import warnings
from functools import wraps
from typing import Optional, Union

import pandas as pd

from pandera.api.pyspark.container import DataFrameSchema

"""Register pandas accessor for pandera schema metadata."""




Schemas = Union[DataFrameSchema]


# Todo Refactor to create a seperate module for panderaAccessor
class PanderaAccessor:
    """Pandera accessor for pandas object."""

    def __init__(self, pandas_obj):
        """Initialize the pandera accessor."""
        self._pandas_obj = pandas_obj
        self._schema: Optional[Schemas] = None

    @staticmethod
    def check_schema_type(schema: Schemas):
        """Abstract method for checking the schema type."""
        raise NotImplementedError

    def add_schema(self, schema):
        """Add a schema to the pandas object."""
        self.check_schema_type(schema)
        self._schema = schema
        return self._pandas_obj

    @property
    def schema(self) -> Optional[Schemas]:
        """Access schema metadata."""
        return self._schema


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
    from pyspark.sql import DataFrame

    return _register_accessor(name, DataFrame)


# def register_series_accessor(name):
#     """
#     Register a custom accessor with a Series object
#
#     :param name: name used when calling the accessor after its registered
#     :returns: a callable class decorator
#     """
#     # pylint: disable=import-outside-toplevel
#
#     from pyspark.sql.functions import col
#
#     return _register_accessor(name, col)
#


class PanderaDataFrameAccessor(PanderaAccessor):
    """Pandera accessor for pandas DataFrame."""

    @staticmethod
    def check_schema_type(schema):
        if not isinstance(schema, DataFrameSchema):
            raise TypeError(
                f"schema arg must be a DataFrameSchema, found {type(schema)}"
            )


# @register_series_accessor("pandera")
# class PanderaSeriesAccessor(PanderaAccessor):
#     """Pandera accessor for pandas Series."""
#
#     @staticmethod
#     def check_schema_type(schema):
#         if not isinstance(schema, SeriesSchema):
#             raise TypeError(
#                 f"schema arg must be a SeriesSchema, found {type(schema)}"
#             )


register_dataframe_accessor("pandera")(PanderaDataFrameAccessor)
# register_series_accessor("pandera")(PanderaSeriesAccessor)
