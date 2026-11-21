# pylint: skip-file
"""Mypy tests for DataFrame instance attribute access."""

import pandera.pandas as pa
from pandera.typing import DataFrame, Series


class Schema(pa.DataFrameModel):
    id: Series[int]
    name: Series[str]


def fn_attr_id(df: DataFrame[Schema]) -> Series[int]:
    return df.id  # mypy okay?


def fn_attr_name(df: DataFrame[Schema]) -> Series[str]:
    return df.name  # mypy okay?


def fn_attr_invalid(df: DataFrame[Schema]) -> Series[str]:
    return df.id  # mypy error
    # error: Incompatible return value type (got "Series[int]", expected "Series[str]")  [return-value]
