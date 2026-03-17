"""Register pandas accessor for pandera schema metadata."""

from typing import Union

import pandas as pd

from pandera.accessors._schema_registry import get_schema, register_schema
from pandera.api.pandas.array import SeriesSchema
from pandera.api.pandas.container import DataFrameSchema

Schemas = Union[DataFrameSchema, SeriesSchema]

_ATTRS_SCHEMA_KEY = "__pandera_schema__"


class PanderaAccessor:
    """Pandera accessor for pandas object."""

    def __init__(self, pandas_obj):
        """Initialize the pandera accessor."""
        self._pandas_obj = pandas_obj

    @staticmethod
    def check_schema_type(schema: Schemas):
        """Abstract method for checking the schema type."""
        raise NotImplementedError

    def add_schema(self, schema):
        """Add a schema to the pandas object."""
        self.check_schema_type(schema)
        if hasattr(self._pandas_obj, "attrs"):
            self._pandas_obj.attrs.pop(_ATTRS_SCHEMA_KEY, None)
        register_schema(self._pandas_obj, schema)
        return self._pandas_obj

    @property
    def schema(self) -> Schemas | None:
        """Access schema metadata."""
        return get_schema(self._pandas_obj)


@pd.api.extensions.register_dataframe_accessor("pandera")
class PanderaDataFrameAccessor(PanderaAccessor):
    """Pandera accessor for pandas DataFrame."""

    @staticmethod
    def check_schema_type(schema):
        if not isinstance(schema, DataFrameSchema):
            raise TypeError(
                f"schema arg must be a {DataFrameSchema}, found {type(schema)}"
            )


@pd.api.extensions.register_series_accessor("pandera")
class PanderaSeriesAccessor(PanderaAccessor):
    """Pandera accessor for pandas Series."""

    @staticmethod
    def check_schema_type(schema):
        if not isinstance(schema, SeriesSchema):
            raise TypeError(
                f"schema arg must be a {SeriesSchema}, found {type(schema)}"
            )
