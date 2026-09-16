"""Plain DataFrame[Schema] does not infer column dtypes in Pyright."""

import pandas as pd
from typing_extensions import reveal_type

import pandera.pandas as pa
from pandera.typing import DataFrame, Series


class InputSchema(pa.DataFrameModel):
    year: Series[int]
    month: Series[int]


def column_getitem(df: DataFrame[InputSchema]) -> None:
    reveal_type(df["year"])
    reveal_type(df.year)


raw = pd.DataFrame({"year": [2001], "month": [3]})
typed_df = InputSchema.validate(raw)
column_getitem(typed_df)
