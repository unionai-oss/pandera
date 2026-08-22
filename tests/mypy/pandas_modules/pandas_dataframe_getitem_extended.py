# pylint: skip-file
"""Mypy tests for inherited schema columns and attribute access."""

import pandera.pandas as pa
from pandera.typing import DataFrame, Series


class BaseSchema(pa.DataFrameModel):
    id: Series[int]


class ChildSchema(BaseSchema):
    name: Series[str]


class BareSchema(pa.DataFrameModel):
    label: str


def fn_base_col(df: DataFrame[BaseSchema]) -> Series[int]:
    return df["id"]  # mypy okay


def fn_child_col(df: DataFrame[ChildSchema]) -> Series[str]:
    return df["name"]  # mypy okay


def fn_child_inherited(df: DataFrame[ChildSchema]) -> Series[int]:
    return df["id"]  # mypy okay


def fn_bare_col(df: DataFrame[BareSchema]) -> Series[str]:
    return df["label"]  # mypy okay


def fn_invalid_bare(df: DataFrame[BareSchema]) -> Series[int]:
    return df["label"]  # mypy error
    # error: Incompatible return value type (got "Series[str]", expected "Series[int]")  [return-value]
