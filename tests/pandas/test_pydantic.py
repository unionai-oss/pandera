"""Unit tests for pydantic compatibility."""

# pylint:disable=too-few-public-methods,missing-class-docstring
from typing import Optional

import pandas as pd
import pytest

import pandera.pandas as pa
from pandera.engines import pydantic_version
from pandera.typing import DataFrame, Series

try:
    from pydantic import BaseModel, ValidationError
except ImportError:
    pytest.skip("Pydantic not available", allow_module_level=True)


PYDANTIC_V2 = False
if pydantic_version().release >= (2, 0, 0):
    PYDANTIC_V2 = True


class SimpleSchema(pa.DataFrameModel):
    """Test DataFrameModel."""

    str_col: Series[str] = pa.Field(unique=True)


class TypedDfPydantic(BaseModel):
    """Test pydantic model with typed dataframe."""

    df: DataFrame[SimpleSchema]


class DataFrameModelPydantic(BaseModel):
    """Test pydantic model with a DataFrameModel."""

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


@pytest.mark.skipif(
    not PYDANTIC_V2,
    reason="Pydantic <2 cannot catch the invalid dataframe model error",
)
def test_invalid_typed_dataframe():
    """Test that an invalid typed DataFrame is recognized by pandera."""
    with pytest.raises(ValidationError):
        TypedDfPydantic(df=1)

    class InvalidSchema(pa.DataFrameModel):
        """Test DataFrameModel."""

        str_col = pa.Field(unique=True)  # omit annotation

    with pytest.raises(pa.errors.SchemaInitError):

        class PydanticModel(BaseModel):
            pa_schema: DataFrame[InvalidSchema]

    # This check prevents Linters from raising an error about not using the PydanticModel class
    with pytest.raises(UnboundLocalError):
        PydanticModel(pa_schema=InvalidSchema)


def test_dataframemodel():
    """Test that DataFrameModel is compatible with pydantic."""
    assert isinstance(
        DataFrameModelPydantic(pa_schema=SimpleSchema),
        DataFrameModelPydantic,
    )


def test_invalid_dataframemodel():
    """Test that an invalid typed DataFrameModel is recognized by pydantic."""
    with pytest.raises(TypeError if PYDANTIC_V2 else ValidationError):
        DataFrameModelPydantic(pa_schema=1)

    with pytest.raises(TypeError if PYDANTIC_V2 else ValidationError):
        DataFrameModelPydantic(pa_schema=SimpleSchema.to_schema())

    class InvalidSchema(pa.DataFrameModel):
        """Test SchemaDataFrameModelModel."""

        str_col = pa.Field(unique=True)  # omit annotation

    class PydanticModel(BaseModel):
        pa_schema: InvalidSchema

    with pytest.raises(ValidationError):
        PydanticModel(pa_schema=InvalidSchema)


def test_dataframemodel_inheritance():
    """Test that an inherited DataFrameModel is compatible with pydantic."""

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
    with pytest.raises(TypeError if PYDANTIC_V2 else ValidationError):
        DataFrameSchemaPydantic(pa_schema=1)

    with pytest.raises(TypeError if PYDANTIC_V2 else ValidationError):
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
    with pytest.raises(TypeError if PYDANTIC_V2 else ValidationError):
        SeriesSchemaPydantic(pa_series_schema=1)

    with pytest.raises(TypeError if PYDANTIC_V2 else ValidationError):
        SeriesSchemaPydantic(pa_column="1")

    with pytest.raises(TypeError if PYDANTIC_V2 else ValidationError):
        SeriesSchemaPydantic(pa_index="1")


@pytest.mark.parametrize(
    "col_type,dtype,item",
    [
        (pa.STRING, "string", "hello"),
        (pa.UINT8, "UInt8", 1),
        (pa.INT8, "Int8", 1),
        (pa.BOOL, "boolean", True),
    ],
)
def test_model_with_extensiondtype_column(col_type, dtype, item):
    """Test that a model with an external dtype is recognized by pydantic."""

    class ExtensionDtypeModel(pa.DataFrameModel):
        a: Series[col_type]

    class PydanticModel(BaseModel):
        df: DataFrame[ExtensionDtypeModel]

    assert isinstance(
        PydanticModel(
            df=DataFrame[ExtensionDtypeModel](
                pd.DataFrame({"a": [item]}, dtype=dtype)
            )
        ),
        PydanticModel,
    )
