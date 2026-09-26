"""Unit tests for pydantic compatibility."""

from typing import (
    Annotated,
    Generic,
    Optional,
    TypeVar,
)

import pandas as pd
import pyarrow
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
    import pydantic_core
    from packaging import version
    from pydantic import ConfigDict


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

    pa_schema: pa.DataFrameSchema | None
    pa_mi: pa.MultiIndex | None


class SeriesSchemaPydantic(BaseModel):
    """Test pydantic model with a SeriesSchema, Column and Index."""

    pa_series_schema: pa.SeriesSchema | None
    pa_column: pa.Column | None
    pa_index: pa.Index | None


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


def _json_schema_of(dtype) -> dict:
    """Build a pydantic model around ``DataFrame[Schema]`` for one column dtype
    and return the generated json schema of that column."""

    class OneColumnSchema(pa.DataFrameModel):
        col: Series[dtype]

    class OneColumnModel(BaseModel):
        df: DataFrame[OneColumnSchema]

    return OneColumnModel.model_json_schema()["properties"]["df"]["items"][
        "properties"
    ]["col"]


# pandas' ``build_table_schema`` reports any dtype it cannot express as a
# json-schema type by its dtype string, and ``to_json_schema`` passes that
# string through as the column's "items.type". None of these labels is a key of
# the core-schema map in ``DataFrame.__get_pydantic_core_schema__``.
UNMAPPED_COLUMN_TYPES = [
    pytest.param(
        Annotated[pd.ArrowDtype, pyarrow.binary()], id="arrow-binary"
    ),
    pytest.param(
        Annotated[pd.ArrowDtype, pyarrow.binary(4)],
        id="arrow-fixed-size-binary",
    ),
    pytest.param(
        Annotated[pd.ArrowDtype, pyarrow.list_(pyarrow.int32())],
        id="arrow-list",
    ),
    pytest.param(
        Annotated[pd.ArrowDtype, pyarrow.struct([("a", pyarrow.int32())])],
        id="arrow-struct",
    ),
    pytest.param(
        Annotated[pd.ArrowDtype, pyarrow.date32()], id="arrow-date32"
    ),
    pytest.param(
        Annotated[pd.ArrowDtype, pyarrow.date64()], id="arrow-date64"
    ),
    pytest.param(
        Annotated[pd.ArrowDtype, pyarrow.time64("us")], id="arrow-time64"
    ),
    pytest.param(
        Annotated[
            pd.ArrowDtype,
            pyarrow.dictionary(pyarrow.int32(), pyarrow.string()),
        ],
        id="arrow-dictionary",
    ),
    pytest.param(
        Annotated[
            pd.ArrowDtype, pyarrow.map_(pyarrow.string(), pyarrow.string())
        ],
        id="arrow-map",
    ),
    pytest.param("interval[int64]", id="pandas-interval"),
    # the label embeds a temporal *field*, but the column is a struct
    pytest.param(
        Annotated[
            pd.ArrowDtype, pyarrow.struct([("t", pyarrow.timestamp("ns"))])
        ],
        id="arrow-struct-of-timestamp",
    ),
]

ARROW_TEMPORAL_FORMATS = [
    pytest.param(
        Annotated[pd.ArrowDtype, pyarrow.timestamp("ns")],
        "date-time",
        id="timestamp",
    ),
    pytest.param(
        Annotated[pd.ArrowDtype, pyarrow.timestamp("us", "UTC")],
        "date-time",
        id="timestamp-tz",
    ),
    pytest.param(
        Annotated[pd.ArrowDtype, pyarrow.duration("s")],
        "duration",
        id="duration",
    ),
]


@pytest.mark.skipif(
    not PYDANTIC_V2,
    reason="json_schema_input_schema is only built for pydantic v2",
)
@pytest.mark.parametrize("dtype", UNMAPPED_COLUMN_TYPES)
def test_typed_dataframe_with_dtype_outside_json_schema_types(dtype):
    """Regression test for #2016.

    Building a pydantic model around ``DataFrame[Schema]`` raised ``KeyError``
    for every dtype here, because the core-schema lookup treated the column's
    json-schema type as a closed set of labels. Such a column is now left
    untyped instead of breaking the whole model.
    """
    assert "type" not in _json_schema_of(dtype)


@pytest.mark.skipif(
    not PYDANTIC_V2,
    reason="json_schema_input_schema is only built for pydantic v2",
)
@pytest.mark.parametrize("dtype,expected_format", ARROW_TEMPORAL_FORMATS)
def test_arrow_temporal_column_keeps_its_format(dtype, expected_format):
    """Temporal dtypes keep the format their numpy counterpart gets."""
    assert _json_schema_of(dtype)["format"] == expected_format


@pytest.mark.skipif(
    not PYDANTIC_V2,
    reason="json_schema_input_schema is only built for pydantic v2",
)
def test_pyarrow_temporal_column_matches_its_numpy_counterpart():
    """An arrow temporal column should render exactly like its numpy twin."""
    assert _json_schema_of(
        Annotated[pd.ArrowDtype, pyarrow.timestamp("ns")]
    ) == _json_schema_of("datetime64[ns]")
    assert _json_schema_of(
        Annotated[pd.ArrowDtype, pyarrow.duration("s")]
    ) == _json_schema_of("timedelta64[ns]")


@pytest.mark.skipif(
    not PYDANTIC_V2,
    reason="json_schema_input_schema is only built for pydantic v2",
)
def test_typed_dataframe_with_arrow_dtype_still_validates():
    """The fix must not weaken validation for an arrow-backed column."""

    class ArrowSchema(pa.DataFrameModel):
        ts: Series[Annotated[pd.ArrowDtype, pyarrow.timestamp("ns")]]

    class ArrowModel(BaseModel):
        df: DataFrame[ArrowSchema]

    valid = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2026-09-18"]).astype(
                pd.ArrowDtype(pyarrow.timestamp("ns"))
            )
        }
    )
    assert isinstance(ArrowModel(df=valid), ArrowModel)

    with pytest.raises(ValidationError):
        ArrowModel(df=pd.DataFrame({"wrong_name": [1]}))


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
    with pytest.raises(pa.errors.SchemaErrors) as exc_info:
        PydanticSchema.validate(invalid_column_names)
    err_msg = exc_info.value.schema_errors[0].args[0]
    assert "Missing columns" in err_msg
    assert "y" in err_msg and "z" in err_msg


@pytest.mark.skipif(
    not PYDANTIC_V2,
    reason="Pydantic <2 already coerces numbers to strings by default",
)
def test_dataframemodel_with_pydantic_model_coerce_numbers_to_str():
    """Test DataFrameModel validation with explicit pydantic v2 string coercion."""

    class Record(BaseModel):
        model_config = ConfigDict(coerce_numbers_to_str=True)

        name: str
        age: int
        city: str

    class Schema(pa.DataFrameModel):
        class Config:
            dtype = pa.engines.pandas_engine.PydanticModel(Record)
            coerce = True

    data = pd.DataFrame(
        {
            "name": [1, "Bob", "Charlie"],
            "age": [25, 30, 22],
            "city": ["New York", "London", "Paris"],
        }
    )

    validated = Schema.validate(data)
    assert validated.to_dict(orient="list") == {
        "name": ["1", "Bob", "Charlie"],
        "age": [25, 30, 22],
        "city": ["New York", "London", "Paris"],
    }
