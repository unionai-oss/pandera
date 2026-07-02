# pylint: skip-file
import pandera.pandas as pa
from pandera.typing import DataFrame, Series


class Schema(pa.DataFrameModel):
    id: Series[int]


def fn_ok(df: DataFrame[Schema]) -> Series[int]:
    return df["id"]  # mypy okay


def fn_error(df: DataFrame[Schema]) -> Series[str]:
    return df["id"]  # mypy error
    # error: Incompatible return value type (got "pandera.typing.pandas.Series[int]", expected "pandera.typing.pandas.Series[str]")  [return-value]
