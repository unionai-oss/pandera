"""Unit tests for pydantic datatype."""

import pandas as pd
import pytest
from pydantic import BaseModel, Field

import pandera.pandas as pa
from pandera.api.pandas.array import ArraySchema
from pandera.engines import pydantic_version
from pandera.engines.pandas_engine import PydanticModel

PYDANTIC_V2 = pydantic_version().release >= (2, 0, 0)
if PYDANTIC_V2:
    from pydantic import ConfigDict


class Record(BaseModel):
    """Pydantic record model."""

    name: str
    xcoord: int
    ycoord: int


class PydanticSchema(pa.DataFrameModel):
    """Pandera schema using the pydantic model."""

    class Config:
        """Config with dataframe-level data type."""

        dtype = PydanticModel(Record)


class PanderaSchema(pa.DataFrameModel):
    """Pandera schema that's equivalent to PydanticSchema."""

    name: pa.typing.Series[str]
    xcoord: pa.typing.Series[int]
    ycoord: pa.typing.Series[int]


def test_pydantic_model():
    """Test that pydantic model correctly validates data."""

    @pa.check_types
    def func(df: pa.typing.DataFrame[PydanticSchema]):
        return df

    valid_df = pd.DataFrame(
        {
            "name": ["foo", "bar", "baz"],
            "xcoord": [1.0, 2, 3],
            "ycoord": [4, 5.0, 6],
        }
    )

    invalid_df = pd.DataFrame(
        {
            "name": ["foo", "bar", "baz"],
            "xcoord": [1, 2, "c"],
            "ycoord": [4, 5, "d"],
        }
    )

    validated = func(valid_df)
    PanderaSchema.validate(validated)

    expected_failure_cases = pd.DataFrame(
        {"index": [2], "failure_case": ["{'xcoord': 'c', 'ycoord': 'd'}"]}
    )

    try:
        func(invalid_df)
    except pa.errors.SchemaErrors as exc:
        pd.testing.assert_frame_equal(
            exc.schema_errors[0].failure_cases, expected_failure_cases
        )


@pytest.mark.parametrize("series_type", [pa.SeriesSchema, pa.Column, pa.Index])
def test_pydantic_model_init_errors(series_type: type[ArraySchema]):
    """
    Should raise SchemaInitError with PydanticModel as `SeriesSchemaBase.dtype`
    """
    with pytest.raises(pa.errors.SchemaInitError):
        series_type(dtype=PydanticModel(Record))


@pytest.mark.parametrize("coerce", [True, False])
def test_pydantic_model_coerce(coerce: bool):
    """Test that DataFrameSchema.coerce is always True with pydantic model"""

    dataframe_schema = pa.DataFrameSchema(
        dtype=PydanticModel(Record), coerce=coerce
    )
    assert dataframe_schema.coerce is True


@pytest.mark.skipif(
    not PYDANTIC_V2,
    reason="Pydantic <2 already coerces numbers to strings by default",
)
def test_pydantic_model_coerce_numbers_to_str():
    """Test that pydantic v2 string coercion can be enabled explicitly."""

    class Row(BaseModel):
        model_config = ConfigDict(coerce_numbers_to_str=True)

        name: str
        age: int
        city: str

    schema = pa.DataFrameSchema(dtype=PydanticModel(Row), coerce=True)
    data = pd.DataFrame(
        {
            "name": [1, "Bob", "Charlie"],
            "age": [25, 30, 22],
            "city": ["New York", "London", "Paris"],
        }
    )

    validated = schema.validate(data)
    assert validated.to_dict(orient="list") == {
        "name": ["1", "Bob", "Charlie"],
        "age": [25, 30, 22],
        "city": ["New York", "London", "Paris"],
    }


def test_pydantic_model_preserves_field_aliases_with_strict_schema():
    """Strict schemas should accept and preserve pydantic field aliases."""

    class Row(BaseModel):
        name: str = Field(alias="Name")
        amount: float = Field(alias="Amount in local currency")

    schema = pa.DataFrameSchema(
        dtype=PydanticModel(Row),
        coerce=True,
        strict=True,
    )
    data = pd.DataFrame(
        {
            "Name": ["foo", "bar"],
            "Amount in local currency": [1.32, 3.34],
        }
    )

    validated = schema.validate(data)
    assert validated.columns.tolist() == [
        "Name",
        "Amount in local currency",
    ]
    pd.testing.assert_frame_equal(validated, data)


def test_pydantic_model_validates_empty_dataframe_with_aliases():
    """Empty dataframes should validate against aliased pydantic fields."""

    class Row(BaseModel):
        name: str = Field(alias="Name")
        amount: float = Field(alias="Amount in local currency")

    schema = pa.DataFrameSchema(dtype=PydanticModel(Row), coerce=True, strict=True)
    data = pd.DataFrame(columns=["Name", "Amount in local currency"])
    validated = schema.validate(data)
    assert validated.columns.tolist() == [
        "Name",
        "Amount in local currency",
    ]
    assert validated.empty
