# pylint: skip-file
"""Regression coverage for Polars DataFrameModel class attribute column names."""

import pandera.polars as pa
from pandera.typing.polars import Series


class SchemaWithSeries(pa.DataFrameModel):
    a: Series[int]
    b: Series[float]


class SchemaWithBareTypes(pa.DataFrameModel):
    x: int
    y: float


def print_string(string: str):
    """Function that expects a string argument."""
    print(f"type: {type(string)}, value: {string}")


# These should all work without mypy errors when the plugin is enabled
series_columns: list[str] = [SchemaWithSeries.a, SchemaWithSeries.b]
bare_columns: list[str] = [SchemaWithBareTypes.x, SchemaWithBareTypes.y]

# These should work with the plugin enabled
print_string(SchemaWithBareTypes.x)
print_string(SchemaWithBareTypes.y)
print_string(SchemaWithSeries.a)
print_string(SchemaWithSeries.b)
