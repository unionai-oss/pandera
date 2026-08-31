"""check_types and validate preserve output schema types."""

import pandas as pd
from typing_extensions import reveal_type

import pandera.pandas as pa
from pandera.typing import DataFrame, Series


class InputSchema(pa.DataFrameModel):
    year: Series[int]
    month: Series[int]
    day: Series[int]


class OutputSchema(InputSchema):
    revenue: Series[float]


@pa.check_types
def transform(df: DataFrame[InputSchema]) -> DataFrame[OutputSchema]:
    return df.assign(revenue=100.0).pipe(DataFrame[OutputSchema])


raw = pd.DataFrame({"year": [2001], "month": [3], "day": [200]})
result = transform(raw)  # pyright: ignore[reportArgumentType]
reveal_type(result)

validated = InputSchema.validate(raw)
reveal_type(validated)

typed_df: DataFrame[InputSchema] = InputSchema.validate(raw)
reveal_type(typed_df)
