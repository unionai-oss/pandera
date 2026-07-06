"""Typed dataframe wrapper enables column inference in Pyright/Pylance."""

from typing import Literal, cast, overload

from typing_extensions import reveal_type

import pandera.pandas as pa
from pandera.typing import DataFrame, Series


class InputSchema(pa.DataFrameModel):
    year: Series[int]
    month: Series[int]
    day: Series[int]


class InputSchemaDataFrame(DataFrame[InputSchema]):
    """Typed dataframe wrapper for static analysis (Pyright/Pylance)."""

    year: Series[int]
    month: Series[int]
    day: Series[int]

    @overload
    def __getitem__(self, key: Literal["year"]) -> Series[int]: ...
    @overload
    def __getitem__(self, key: Literal["month"]) -> Series[int]: ...
    @overload
    def __getitem__(self, key: Literal["day"]) -> Series[int]: ...
    @overload
    def __getitem__(self, key: str) -> Series: ...
    def __getitem__(self, key: str) -> Series:
        return cast(Series, super().__getitem__(key))


@pa.check_types
def transform(df: InputSchemaDataFrame) -> InputSchemaDataFrame:
    reveal_type(df)
    reveal_type(df["year"])
    return df


def typed_getitem(df: InputSchemaDataFrame) -> Series[int]:
    reveal_type(df["year"])
    return df["year"]
