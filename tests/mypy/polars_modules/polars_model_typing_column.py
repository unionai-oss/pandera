"""No-plugin static coverage for ``pandera.typing.Column``."""

import polars as pl

import pandera.polars as pa
from pandera.typing import Column


class Schema(pa.DataFrameModel):
    required: Column[pl.List]
    nullable_values: Column[int | None]
    optional_presence: Column[int] | None


def accepts_name(name: str) -> str:
    return name


required_name: str = Schema.required
nullable_name: str = Schema.nullable_values
required_names: list[str] = [Schema.required, Schema.nullable_values]
accepted_name: str = accepts_name(Schema.required)
optional_name: str | None = Schema.optional_presence
