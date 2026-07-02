"""Issue #1487: plain DataFrame[Schema] does not infer column dtypes in Pyright."""

import pandera.pandas as pa
from pandera.typing import DataFrame, Series


class InputSchema(pa.DataFrameModel):
    year: Series[int]
    month: Series[int]


def plain_getitem(df: DataFrame[InputSchema]) -> Series[int]:
    return df["year"]


def plain_attr(df: DataFrame[InputSchema]) -> Series[int]:
    return df.year
