"""Custom accessor functionality for PySpark.Sql. Register pyspark accessor for pandera schema metadata.
"""

import warnings
from typing import Optional

from pandera.api.base.error_handler import ErrorHandler
from pandera.api.pyspark.container import DataFrameSchema

Schemas = DataFrameSchema  # type: ignore
Errors = ErrorHandler  # type: ignore


class PanderaAccessor:
    """Pandera accessor for pyspark object."""

    def __init__(self, pyspark_obj):
        """Initialize the pandera accessor."""
        self._pyspark_obj = pyspark_obj
        self._schema: Optional[Schemas] = None
        self._errors: Optional[Errors] = None

    @staticmethod
    def check_schema_type(schema: Schemas):  # type: ignore
        """Abstract method for checking the schema type."""
        raise NotImplementedError

    def add_schema(self, schema):
        """Add a schema to the pyspark object."""
        self.check_schema_type(schema)
        self._schema = schema
        return self._pyspark_obj

    @property
    def schema(self) -> Optional[Schemas]:  # type: ignore
        """Access schema metadata."""
        return self._schema

    @property
    def errors(self) -> Optional[Errors]:  # type: ignore
        """Access errors details."""
        return self._errors

    @errors.setter
    def errors(self, value: Optional[Errors]):  # type: ignore
        """Set errors details."""
        self._errors = value


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


class PanderaDataFrameAccessor(PanderaAccessor):
    """Pandera accessor for pyspark DataFrame."""

    @staticmethod
    def check_schema_type(schema):
        if not isinstance(schema, DataFrameSchema):
            raise TypeError(
                f"schema arg must be a DataFrameSchema, found {type(schema)}"
            )


register_dataframe_accessor("pandera")(PanderaDataFrameAccessor)
