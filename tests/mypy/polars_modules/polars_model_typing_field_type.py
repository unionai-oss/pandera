"""No-plugin static coverage for ``pandera.typing.FieldType``."""

import polars as pl

import pandera.polars as pa
from pandera.typing import FieldType


class Schema(pa.DataFrameModel):
    required: FieldType[pl.List] = pa.Field()
    nullable_values: FieldType[int | None] = pa.Field()
    optional_presence: FieldType[int] | None
    optional_assignment: FieldType[int] = pa.Field(required=False)


def accepts_name(name: str) -> str:
    return name


required_name: str = Schema.required
nullable_name: str = Schema.nullable_values
required_names: list[str] = [Schema.required, Schema.nullable_values]
accepted_name: str = accepts_name(Schema.required)
optional_name: str | None = Schema.optional_presence
optional_assignment_name: str = Schema.optional_assignment
