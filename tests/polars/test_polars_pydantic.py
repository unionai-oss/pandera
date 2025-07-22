"""Unit tests for Polars pydantic compatibility."""

from typing import Optional

import polars as pl
import pytest

import pandera.polars as pa
from pandera.engines import pydantic_version
from pandera.typing.polars import DataFrame, Series

try:
    from pydantic import BaseModel, ValidationError
except ImportError:
    pytest.skip("Pydantic not available", allow_module_level=True)


PYDANTIC_V2 = False
if pydantic_version().release >= (2, 0, 0):
    PYDANTIC_V2 = True


class SimplePolarsSchema(pa.DataFrameModel):
    """Test Polars DataFrameModel."""

    str_col: Series[str] = pa.Field(unique=True)


class TypedPolarsDataFramePydantic(BaseModel):
    """Test pydantic model with typed polars dataframe."""

    df: DataFrame[SimplePolarsSchema]


class PolarsDataFrameModelPydantic(BaseModel):
    """Test pydantic model with a Polars DataFrameModel."""

    pa_schema: SimplePolarsSchema


class PolarsDataFrameSchemaPydantic(BaseModel):
    """Test pydantic model with a Polars DataFrameSchema."""

    pa_schema: Optional[pa.DataFrameSchema]


def test_typed_polars_dataframe():
    """Test that typed Polars DataFrame is compatible with pydantic."""
    valid_df = pl.DataFrame({"str_col": ["hello", "world"]})
    assert isinstance(
        TypedPolarsDataFramePydantic(df=valid_df), TypedPolarsDataFramePydantic
    )

    invalid_df = pl.DataFrame({"str_col": ["hello", "hello"]})
    with pytest.raises(ValidationError):
        TypedPolarsDataFramePydantic(df=invalid_df)


@pytest.mark.skipif(
    not PYDANTIC_V2,
    reason="Pydantic <2 cannot catch the invalid dataframe model error",
)
def test_invalid_typed_polars_dataframe():
    """Test that an invalid typed Polars DataFrame is recognized by pandera."""
    with pytest.raises(ValidationError):
        TypedPolarsDataFramePydantic(df=1)

    class InvalidSchema(pa.DataFrameModel):
        """Test DataFrameModel."""

        str_col = pa.Field(unique=True)  # omit annotation

    with pytest.raises(pa.errors.SchemaInitError):

        class PydanticModel(BaseModel):
            pa_schema: DataFrame[InvalidSchema]

    # This check prevents Linters from raising an error about not using the PydanticModel class
    with pytest.raises(UnboundLocalError):
        PydanticModel(pa_schema=InvalidSchema)


def test_polars_dataframemodel():
    """Test that Polars DataFrameModel is compatible with pydantic."""
    assert isinstance(
        PolarsDataFrameModelPydantic(pa_schema=SimplePolarsSchema),
        PolarsDataFrameModelPydantic,
    )


def test_invalid_polars_dataframemodel():
    """Test that an invalid typed Polars DataFrameModel is recognized by pydantic."""
    with pytest.raises(TypeError if PYDANTIC_V2 else ValidationError):
        PolarsDataFrameModelPydantic(pa_schema=1)

    with pytest.raises(TypeError if PYDANTIC_V2 else ValidationError):
        PolarsDataFrameModelPydantic(pa_schema=SimplePolarsSchema.to_schema())


def test_polars_dataframemodel_inheritance():
    """Test that an inherited Polars DataFrameModel is compatible with pydantic."""

    class Parent(pa.DataFrameModel):
        a: Series[str]

    class Child(Parent):
        b: Series[str]

    class PydanticModel(BaseModel):
        pa_schema: Parent

    assert isinstance(PydanticModel(pa_schema=Parent), PydanticModel)
    assert isinstance(PydanticModel(pa_schema=Child), PydanticModel)

    class NotChild(pa.DataFrameModel):
        b: Series[str]

    with pytest.raises(TypeError if PYDANTIC_V2 else ValidationError):
        assert isinstance(PydanticModel(pa_schema=NotChild), PydanticModel)


def test_polars_dataframeschema():
    """Test that Polars DataFrameSchema is compatible with pydantic."""
    assert isinstance(
        PolarsDataFrameSchemaPydantic(pa_schema=pa.DataFrameSchema()),
        PolarsDataFrameSchemaPydantic,
    )


def test_optional_column_schema():
    """Test that a Polars DataFrameModel with optional columns works correctly."""

    class SchemaWithOptionalColumn(pa.DataFrameModel):
        """Test Polars DataFrameModel with an optional column."""

        required_col: Series[str]
        optional_col: Optional[Series[int]]

    # Create a custom pydantic model that uses our schema with optional column
    class OptionalColumnModelPydantic(BaseModel):
        df: DataFrame[SchemaWithOptionalColumn]

    # Test with only the required column
    df_required_only = pl.DataFrame({"required_col": ["value1", "value2"]})
    model = OptionalColumnModelPydantic(df=df_required_only)
    assert isinstance(model, OptionalColumnModelPydantic)

    # Test with both required and optional columns
    df_with_optional = pl.DataFrame(
        {"required_col": ["value1", "value2"], "optional_col": [1, 2]}
    )
    model = OptionalColumnModelPydantic(df=df_with_optional)
    assert isinstance(model, OptionalColumnModelPydantic)

    # Test with invalid optional column type
    df_invalid_optional = pl.DataFrame(
        {
            "required_col": ["value1", "value2"],
            "optional_col": ["not_an_int", "invalid"],
        }
    )
    with pytest.raises(ValidationError):
        OptionalColumnModelPydantic(df=df_invalid_optional)

    # Test with missing required column
    df_missing_required = pl.DataFrame({"optional_col": [1, 2]})
    with pytest.raises(ValidationError):
        OptionalColumnModelPydantic(df=df_missing_required)


def test_invalid_polars_dataframeschema():
    """Test that an invalid Polars DataFrameSchema is recognized by pydantic."""
    with pytest.raises(TypeError if PYDANTIC_V2 else ValidationError):
        PolarsDataFrameSchemaPydantic(pa_schema=1)


def test_conversion_types():
    """Test that various input formats can be converted to Polars DataFrames."""
    # Test dict conversion
    data_dict = {"str_col": ["hello", "world"]}
    model = TypedPolarsDataFramePydantic(df=data_dict)
    assert isinstance(model.df, pl.DataFrame)

    # Test with pandas DataFrame
    try:
        import pandas as pd

        pandas_df = pd.DataFrame({"str_col": ["hello", "world"]})
        model = TypedPolarsDataFramePydantic(df=pandas_df)
        assert isinstance(model.df, pl.DataFrame)
    except ImportError:
        pytest.skip("pandas not installed")
