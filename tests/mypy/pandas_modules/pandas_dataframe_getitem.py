# pylint: skip-file
"""Static type checking for DataFrame column access via __getitem__."""

import pandera.pandas as pa
from pandera.typing import DataFrame, Series


class Schema(pa.DataFrameModel):
    id: Series[int]
    name: Series[str]


def fn(df: DataFrame[Schema]) -> Series[int]:
    return df["id"]  # mypy okay


def fn_invalid(df: DataFrame[Schema]) -> Series[str]:
    return df["id"]  # mypy error
    # error: Incompatible return value type (got "Series[int]", expected "Series[str]")  [return-value]


def fn_unknown_column(df: DataFrame[Schema]) -> Series[int]:
    return df["missing"]  # mypy okay - falls back to generic Series
