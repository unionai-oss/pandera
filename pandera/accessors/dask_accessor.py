"""Register dask accessor for pandera schema metadata."""

from typing import Union

from dask.dataframe.extensions import (
    register_dataframe_accessor,
    register_series_accessor,
)

from pandera.api.pandas.array import SeriesSchema
from pandera.api.pandas.container import DataFrameSchema

Schemas = Union[DataFrameSchema, SeriesSchema]


class PanderaDaskAccessor:
    """Pandera accessor for dask objects.

    Unlike pandas DataFrames/Series, Dask objects don't support the `.attrs`
    attribute directly. Instead, we store the schema in the underlying
    `._meta` object which is a pandas DataFrame/Series.
    """

    # Key used to store schema in the meta object's attrs
    _PANDERA_SCHEMA_KEY = "__pandera_schema__"

    def __init__(self, dask_obj):
        """Initialize the pandera accessor."""
        self._dask_obj = dask_obj

    @staticmethod
    def check_schema_type(schema: Schemas):
        """Abstract method for checking the schema type."""
        raise NotImplementedError

    def add_schema(self, schema):
        """Add a schema to the dask object via its _meta attribute."""
        self.check_schema_type(schema)
        # Store schema in the _meta object's attrs (which is a pandas object)
        self._dask_obj._meta.attrs[self._PANDERA_SCHEMA_KEY] = schema
        return self._dask_obj

    @property
    def schema(self) -> Schemas | None:
        """Access schema metadata from the _meta object."""
        return self._dask_obj._meta.attrs.get(self._PANDERA_SCHEMA_KEY)


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
