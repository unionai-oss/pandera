# pylint: skip-file
"""Mypy tests for https://github.com/unionai-oss/pandera/issues/1487.

Covers schema-level type retention and column-level inference with the
pandera mypy plugin enabled.
"""

from typing import cast

import pandas as pd

import pandera.pandas as pa
from pandera.typing import DataFrame, Series


class InputSchema(pa.DataFrameModel):
    year: Series[int]
    month: Series[int]
    day: Series[int]


class OutputSchema(InputSchema):
    revenue: Series[float]


# --- Schema retention after validation / construction ---


def accepts_input(df: DataFrame[InputSchema]) -> None: ...


def accepts_output(df: DataFrame[OutputSchema]) -> None: ...


raw = pd.DataFrame({"year": [2001], "month": [3], "day": [200]})

validated = InputSchema.validate(raw)
constructed = cast(DataFrame[InputSchema], InputSchema(raw))
typed_construction = DataFrame[InputSchema](raw)

accepts_input(validated)  # mypy okay
accepts_input(constructed)  # mypy okay
accepts_input(typed_construction)  # mypy okay

accepts_input(raw)  # mypy error
# error: Argument 1 to "accepts_input" has incompatible type "pandas.core.frame.DataFrame";  # noqa
# expected "pandera.typing.pandas.DataFrame[InputSchema]"  [arg-type]


# --- Column access via __getitem__ ---


def get_year_column(df: DataFrame[InputSchema]) -> Series[int]:
    return df["year"]  # mypy okay


def get_year_wrong_type(df: DataFrame[InputSchema]) -> Series[str]:
    return df["year"]  # mypy error
    # error: Incompatible return value type (got "Series[int]", expected "Series[str]")  [return-value]


def get_unknown_column(df: DataFrame[InputSchema]) -> Series[int]:
    return df["missing"]  # mypy okay


# --- Column access via attribute ---


def get_year_attr(df: DataFrame[InputSchema]) -> Series[int]:
    return df.year  # mypy okay


def get_year_attr_wrong_type(df: DataFrame[InputSchema]) -> Series[str]:
    return df.year  # mypy error
    # error: Incompatible return value type (got "Series[int]", expected "Series[str]")  [return-value]


# --- check_types preserves schema ---


@pa.check_types
def transform(df: DataFrame[InputSchema]) -> DataFrame[OutputSchema]:
    return df.assign(revenue=100.0).pipe(DataFrame[OutputSchema])  # mypy okay


@pa.check_types
def transform_input(df: DataFrame[InputSchema]) -> DataFrame[InputSchema]:
    return df  # mypy okay


typed_df = transform_input(DataFrame[InputSchema](raw))  # mypy okay
accepts_input(typed_df)  # mypy okay

transformed = transform(DataFrame[InputSchema](raw))  # mypy okay
accepts_output(transformed)  # mypy okay
