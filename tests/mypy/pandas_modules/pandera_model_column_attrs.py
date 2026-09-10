# pylint: skip-file
"""Regression coverage for DataFrameModel class attribute column names."""

import pandera.pandas as pa
from pandera.typing import Series
from pandera.typing import pandas as pdt


class SchemaWithSeries(pa.DataFrameModel):
    a: Series[int]
    b: pdt.Series[float]


class SchemaWithBareTypes(pa.DataFrameModel):
    x: int
    y: float


series_columns: list[str] = [SchemaWithSeries.a, SchemaWithSeries.b]
bare_columns: list[str] = [SchemaWithBareTypes.x, SchemaWithBareTypes.y]
