"""Issue #1487: schema-level types retained after validate/check_types/construction."""

import pandas as pd
from typing_extensions import reveal_type

import pandera.pandas as pa
from pandera.typing import DataFrame, Series


class InputSchema(pa.DataFrameModel):
    year: Series[int]
    month: Series[int]


@pa.check_types
def transform(df: DataFrame[InputSchema]) -> DataFrame[InputSchema]:
    return df


raw = pd.DataFrame({"year": [2001], "month": [3]})

result_check_types = transform(raw)  # pyright: ignore[reportArgumentType]
reveal_type(result_check_types)

result_validate = InputSchema.validate(raw)
reveal_type(result_validate)

result_constructor = InputSchema(raw)
reveal_type(result_constructor)

result_generic = DataFrame[InputSchema](raw)  # pyright: ignore[reportCallIssue]
reveal_type(result_generic)


def accept(df: DataFrame[InputSchema]) -> None:
    reveal_type(df)


accept(result_validate)
accept(result_constructor)
accept(result_generic)
