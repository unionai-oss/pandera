"""Unit tests for pydantic compatibility."""
# pylint:disable=too-few-public-methods,missing-class-docstring
from typing import Optional

import pandas as pd
import pytest

import pandera as pa
from pandera.typing import DataFrame, Series

try:
    from pydantic import BaseModel, ValidationError
except ImportError:
    pytest.skip("Pydantic not available", allow_module_level=True)


class SimpleSchema(pa.SchemaModel):
    """Test SchemaModel."""

    str_col: Series[str] = pa.Field(unique=True)


class TypedDfPydantic(BaseModel):
    """Test pydantic model with typed dataframe."""

    df: DataFrame[SimpleSchema]


class SchemaModelPydantic(BaseModel):
    """Test pydantic model with a SchemaModel."""

    pa_schema: SimpleSchema


class DataFrameSchemaPydantic(BaseModel):
    """Test pydantic model with a DataFrameSchema and MultiIndex."""

    pa_schema: Optional[pa.DataFrameSchema]
    pa_mi: Optional[pa.MultiIndex]


class SeriesSchemaPydantic(BaseModel):
    """Test pydantic model with a SeriesSchema, Column and Index."""

    pa_series_schema: Optional[pa.SeriesSchema]
    pa_column: Optional[pa.Column]
    pa_index: Optional[pa.Index]


def test_typed_dataframe():
    """Test that typed DataFrame is compatible with pydantic."""
    valid_df = pd.DataFrame({"str_col": ["hello", "world"]})
    assert isinstance(TypedDfPydantic(df=valid_df), TypedDfPydantic)

    invalid_df = pd.DataFrame({"str_col": ["hello", "hello"]})
    with pytest.raises(ValidationError):
        TypedDfPydantic(df=invalid_df)


def test_invalid_typed_dataframe():
    """Test that an invalid typed DataFrame is recognized by pydantic."""
    with pytest.raises(ValidationError):
        TypedDfPydantic(df=1)

    class InvalidSchema(pa.SchemaModel):
        """Test SchemaModel."""

        str_col = pa.Field(unique=True)  # omit annotation

    class PydanticModel(BaseModel):
        pa_schema: DataFrame[InvalidSchema]

    with pytest.raises(ValueError):
        PydanticModel(pa_schema=InvalidSchema)


def test_schemamodel():
    """Test that SchemaModel is compatible with pydantic."""
    assert isinstance(
        SchemaModelPydantic(pa_schema=SimpleSchema),
        SchemaModelPydantic,
    )


def test_invalid_schemamodel():
    """Test that an invalid typed SchemaModel is recognized by pydantic."""
    with pytest.raises(ValidationError):
        SchemaModelPydantic(pa_schema=1)

    with pytest.raises(ValidationError):
        SchemaModelPydantic(pa_schema=SimpleSchema.to_schema())

    class InvalidSchema(pa.SchemaModel):
        """Test SchemaModel."""

        str_col = pa.Field(unique=True)  # omit annotation

    class PydanticModel(BaseModel):
        pa_schema: InvalidSchema

    with pytest.raises(ValueError):
        PydanticModel(pa_schema=InvalidSchema)


def test_schemamodel_inheritance():
    """Test that an inherited SchemaModel is compatible with pydantic."""

    class Parent(pa.SchemaModel):
        a: Series[str]

    class Child(Parent):
        b: Series[str]

    class PydanticModel(BaseModel):
        pa_schema: Parent

    assert isinstance(PydanticModel(pa_schema=Parent), PydanticModel)
    assert isinstance(PydanticModel(pa_schema=Child), PydanticModel)

    class NotChild(pa.SchemaModel):
        b: Series[str]

    with pytest.raises(ValidationError):
        assert isinstance(PydanticModel(pa_schema=NotChild), PydanticModel)


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
    with pytest.raises(ValidationError):
        DataFrameSchemaPydantic(pa_schema=1)

    with pytest.raises(ValidationError):
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
    with pytest.raises(ValidationError):
        SeriesSchemaPydantic(pa_series_schema=1)

    with pytest.raises(ValidationError):
        SeriesSchemaPydantic(pa_column="1")

    with pytest.raises(ValidationError):
        SeriesSchemaPydantic(pa_index="1")
