"""Unit tests for pydantic compatibility."""

from typing import Optional
from typing import (
    Generic,
    TypeVar,
)

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
    from packaging import version
    import pydantic_core


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


TableT = TypeVar("TableT", bound=pa.DataFrameModel)


class TypedDfGenericPydantic(BaseModel, Generic[TableT]):
    """Test pydantic model with typed generic dataframe."""

    df: DataFrame[TableT]


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


@pytest.mark.skipif(
    not PYDANTIC_V2,
    reason="Pydantic <2 does not use Pydantic-Core",
)
def test_typed_dataframe_model_json_schema():
    """Test that typed generic DataFrame generates model json schema."""

    # pylint: disable-next=possibly-used-before-assignment
    if version.parse(pydantic_core.__version__).release >= (
        2,
        30,
        0,
    ):
        assert isinstance(TypedDfPydantic.model_json_schema(), dict)


@pytest.mark.skipif(
    not PYDANTIC_V2,
    reason="Pydantic <2 cannot catch the invalid dataframe validation error",
)
def test_typed_generic_dataframe():
    """Test that typed generic DataFrame is compatible with pydantic."""
    valid_df = pd.DataFrame({"str_col": ["hello", "world"]})
    TypedDfGenericPydantic[SimpleSchema](df=valid_df)

    invalid_df = pd.DataFrame({"str_col": ["hello", "hello"]})
    with pytest.raises(ValidationError):
        TypedDfGenericPydantic[SimpleSchema](df=invalid_df)


@pytest.mark.skipif(
    not PYDANTIC_V2,
    reason="Pydantic <2 does not use Pydantic-Core",
)
def test_typed_generic_dataframe_model_json_schema():
    """Test that typed generic DataFrame generates model json schema."""

    # pylint: disable-next=possibly-used-before-assignment
    if version.parse(pydantic_core.__version__).release >= (
        2,
        30,
        0,
    ):
        assert isinstance(
            TypedDfGenericPydantic[SimpleSchema].model_json_schema(), dict
        )


def test_pydantic_model_empty_dataframe():
    """
    Test that a Schema with a PydanticModel can validate an empty dataframe,
    but warns the user that no type checking is performed.
    """

    from pandera.engines.pandas_engine import PydanticModel

    class Record(BaseModel):
        x: str
        y: int
        z: float

    class PydanticSchema(pa.DataFrameModel):
        """Pandera schema using the pydantic model."""

        class Config:
            """Config with dataframe-level data type."""

            dtype = PydanticModel(Record)

    if PYDANTIC_V2:
        column_types = {
            col: field_info.annotation
            for col, field_info in Record.model_fields.items()
        }
    else:
        column_types = {
            col: field_info.annotation
            for col, field_info in Record.__fields__.items()
        }

    columns = [*column_types]
    empty_df = pd.DataFrame(columns=columns).astype(column_types)
    with pytest.warns(
        UserWarning, match="PydanticModel cannot validate an empty dataframe"
    ):
        PydanticSchema.validate(empty_df)

    invalid_column_names = pd.DataFrame(columns=columns[:1])
    with pytest.raises(
        pa.errors.SchemaError,
        match=".+Missing columns in .+data_container.+ ['y', 'z'].+",
    ):
        PydanticSchema.validate(invalid_column_names)
