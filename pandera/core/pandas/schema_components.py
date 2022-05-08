import traceback

from pandera.core.pandas.schemas import DataFrameSchema, SeriesSchema
from pandera.error_formatters import scalar_failure_case
from pandera.errors import SchemaError, SchemaErrors


class Column(SeriesSchema):
    ...


class Index(SeriesSchema):
    ...


class MultiIndex(DataFrameSchema):
    ...
