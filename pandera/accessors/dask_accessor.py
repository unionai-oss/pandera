"""Register dask accessor for pandera schema metadata."""

from typing import Union

from dask.dataframe.extensions import (
    register_dataframe_accessor,
    register_series_accessor,
)

from pandera.accessors._schema_registry import get_schema, register_schema
from pandera.api.pandas.array import SeriesSchema
from pandera.api.pandas.container import DataFrameSchema

Schemas = Union[DataFrameSchema, SeriesSchema]


class PanderaDaskAccessor:
    """Pandera accessor for dask objects."""

    def __init__(self, dask_obj):
        """Initialize the pandera accessor."""
        self._dask_obj = dask_obj

    @staticmethod
    def check_schema_type(schema: Schemas):
        """Abstract method for checking the schema type."""
        raise NotImplementedError

    def add_schema(self, schema):
        """Add a schema to the dask object."""
        self.check_schema_type(schema)
        register_schema(self._dask_obj, schema)
        return self._dask_obj

    @property
    def schema(self) -> Schemas | None:
        """Access schema metadata."""
        return get_schema(self._dask_obj)


class PanderaDaskDataFrameAccessor(PanderaDaskAccessor):
    """Pandera accessor for dask DataFrame."""

    @staticmethod
    def check_schema_type(schema):
        if not isinstance(schema, DataFrameSchema):
            raise TypeError(
                f"schema arg must be a {DataFrameSchema}, found {type(schema)}"
            )


class PanderaDaskSeriesAccessor(PanderaDaskAccessor):
    """Pandera accessor for dask Series."""

    @staticmethod
    def check_schema_type(schema):
        if not isinstance(schema, SeriesSchema):
            raise TypeError(
                f"schema arg must be a {SeriesSchema}, found {type(schema)}"
            )


register_dataframe_accessor("pandera")(PanderaDaskDataFrameAccessor)
register_series_accessor("pandera")(PanderaDaskSeriesAccessor)
