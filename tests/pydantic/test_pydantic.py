"""Unit tests for pydantic compatibility."""
# pylint:disable=too-few-public-methods,missing-class-docstring
from typing import Optional

import pandas as pd
import pydantic
import pytest

import pandera as pa
from pandera.typing import DataFrame, Series


class SimpleSchema(pa.SchemaModel):
    """Test SchemaModel."""

    str_col: Series[str] = pa.Field(unique=True)


class TypedDfPydantic(pydantic.BaseModel):
    """Test pydantic model with typed dataframe."""

    df: DataFrame[SimpleSchema]


class SchemaModelPydantic(pydantic.BaseModel):
    """Test pydantic model with a SchemaModel."""

    pa_schema: SimpleSchema


class DataFrameSchemaPydantic(pydantic.BaseModel):
    """Test pydantic model with a DataFrameSchema and MultiIndex."""

    pa_schema: Optional[pa.DataFrameSchema]
    pa_mi: Optional[pa.MultiIndex]


class SeriesSchemaPydantic(pydantic.BaseModel):
    """Test pydantic model with a SeriesSchema, Column and Index."""

    pa_series_schema: Optional[pa.SeriesSchema]
    pa_column: Optional[pa.Column]
    pa_index: Optional[pa.Index]


def test_typed_dataframe():
    """Test that typed DataFrame is compatible with pydantic."""
    valid_df = pd.DataFrame({"str_col": ["hello", "world"]})
    assert isinstance(TypedDfPydantic(df=valid_df), TypedDfPydantic)

    invalid_df = pd.DataFrame({"str_col": ["hello", "hello"]})
    with pytest.raises(pydantic.ValidationError):
        TypedDfPydantic(df=invalid_df)


def test_invalid_typed_dataframe():
    """Test that an invalid typed DataFrame is recognized by pydantic."""
    with pytest.raises(pydantic.ValidationError):
        TypedDfPydantic(df=1)


def test_schemamodel():
    """Test that SchemaModel is compatible with pydantic."""
    assert isinstance(
        SchemaModelPydantic(pa_schema=SimpleSchema),
        SchemaModelPydantic,
    )


def test_invalid_schemamodel():
    """Test that an invalid typed SchemaModel is recognized by pydantic."""
    with pytest.raises(pydantic.ValidationError):
        SchemaModelPydantic(pa_schema=1)

    with pytest.raises(pydantic.ValidationError):
        SchemaModelPydantic(pa_schema=SimpleSchema.to_schema())


def test_schemamodel_inheritance():
    """Test that an inherited SchemaModel is compatible with pydantic."""

    class _Parent(pa.SchemaModel):
        a: Series[str]

    class _Child(_Parent):
        b: Series[str]

    class PydanticModel(pydantic.BaseModel):
        pa_schema: _Parent

    assert isinstance(PydanticModel(pa_schema=_Parent), PydanticModel)
    assert isinstance(PydanticModel(pa_schema=_Child), PydanticModel)


def test_dataframeschema():
    """Test that DataFrameSchema is compatible with pydantic."""
    assert isinstance(
        DataFrameSchemaPydantic(
            pa_schema=pa.DataFrameSchema(),
            pa_mi=pa.MultiIndex([pa.Index(str), pa.Index(int)]),
        ),
        DataFrameSchemaPydantic,
    )


def test_invalid_dataframeschema():
    """Test that an invalid DataFrameSchema is recognized by pydantic."""
    with pytest.raises(pydantic.ValidationError):
        DataFrameSchemaPydantic(pa_schema=1)

    with pytest.raises(pydantic.ValidationError):
        DataFrameSchemaPydantic(pa_mi="1")


def test_seriesschema():
    """Test that SeriesSchemaBase is compatible with pydantic."""
    assert isinstance(
        SeriesSchemaPydantic(
            pa_series_schema=pa.SeriesSchema(),
            pa_column=pa.Column(),
            pa_index=pa.Index(),
        ),
        SeriesSchemaPydantic,
    )


def test_invalid_seriesschema():
    """Test that an invalid SeriesSchemaBase is recognized by pydantic."""
    with pytest.raises(pydantic.ValidationError):
        DataFrameSchemaPydantic(pa_series_schema=1)

    with pytest.raises(pydantic.ValidationError):
        DataFrameSchemaPydantic(pa_column="1")

    with pytest.raises(pydantic.ValidationError):
        DataFrameSchemaPydantic(pa_index="1")
