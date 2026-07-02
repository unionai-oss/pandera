"""Tests for typed dataframe source generation."""

from __future__ import annotations

import pandera.pandas as pa
from pandera.typing import Series
from pandera.typing.codegen import generate_typed_dataframe_source


class SampleSchema(pa.DataFrameModel):
    id: Series[int]
    name: Series[str]


def test_generate_typed_dataframe_source_contains_overloads() -> None:
    source = generate_typed_dataframe_source(SampleSchema)
    assert "class SampleSchemaDataFrame(DataFrame[SampleSchema])" in source
    assert "def __getitem__(self, key: Literal['id'])" in source
    assert "-> Series[int]" in source
    assert "id: Series[int]" in source
    assert "cast(Series, super().__getitem__(key))" in source


def test_generate_typed_dataframe_source_custom_name() -> None:
    source = generate_typed_dataframe_source(
        SampleSchema, class_name="TypedSample"
    )
    assert "class TypedSample(DataFrame[SampleSchema])" in source
