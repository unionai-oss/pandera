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


class OptionalFieldModel(BaseModel):
    """Pydantic model with a required field and an optional field."""

    title: str
    rating: int | None = None


class OptionalFieldSchema(pa.DataFrameModel):
    """Pandera schema using a pydantic model with an optional field."""

    class Config:
        dtype = PydanticModel(OptionalFieldModel)


def test_pydantic_model_optional_field_missing_column():
    """
    An optional pydantic field should not be required as a dataframe column.

    Regression test for https://github.com/unionai-oss/pandera/issues/2406
    """
    data = pd.DataFrame({"title": ["Dune", "Foundation"]})
    validated = OptionalFieldSchema.validate(data)
    assert validated["rating"].isna().all()


def test_pydantic_model_optional_field_null_values():
    """
    An optional pydantic field should tolerate null values in the column.

    Regression test for https://github.com/unionai-oss/pandera/issues/2406
    """
    data = pd.DataFrame(
        {"title": ["Dune", "Foundation"], "rating": [None, None]}
    )
    validated = OptionalFieldSchema.validate(data)
    assert validated["rating"].isna().all()


def test_pydantic_model_column_nullability_matches_field_requiredness():
    """
    The auto-generated Column for a required pydantic field must be
    non-nullable, and for an optional field must be nullable.

    This checks the nullable flag directly rather than going through
    validate(), so it holds regardless of how permissively a given
    pydantic version coerces values during row-level parsing (see
    test_pydantic_model_required_field_still_rejects_nulls below).
    """
    schema = OptionalFieldSchema.to_schema()
    assert schema.columns["title"].nullable is False
    assert schema.columns["rating"].nullable is True


@pytest.mark.skipif(
    not PYDANTIC_V2,
    reason=(
        "pydantic v1's `str` validator coerces a non-string, non-null "
        "input like float('nan') into the string 'nan' instead of "
        "rejecting it, so a null value in a required str column never "
        "reaches PydanticModel.coerce()'s ValidationError handling on "
        "v1. This is a pre-existing v1/v2 coercion strictness "
        "difference in PydanticModel, not specific to column "
        "nullability - test_pydantic_model_column_nullability_matches_"
        "field_requiredness above verifies the nullable flag itself on "
        "both versions."
    ),
)
def test_pydantic_model_required_field_still_rejects_nulls():
    """A required pydantic field should still reject null values."""
    data = pd.DataFrame(
        {"title": [None, "Foundation"], "rating": [3, 4]}
    )
    with pytest.raises(pa.errors.SchemaErrors):
        OptionalFieldSchema.validate(data, lazy=True)
