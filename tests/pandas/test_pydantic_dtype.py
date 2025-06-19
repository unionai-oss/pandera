"""Unit tests for pydantic datatype."""

from typing import Type

import pandas as pd
import pytest
from pydantic import BaseModel

import pandera.pandas as pa
from pandera.api.pandas.array import ArraySchema
from pandera.engines.pandas_engine import PydanticModel


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
    except pa.errors.SchemaError as exc:
        pd.testing.assert_frame_equal(
            exc.failure_cases, expected_failure_cases
        )


@pytest.mark.parametrize("series_type", [pa.SeriesSchema, pa.Column, pa.Index])
def test_pydantic_model_init_errors(series_type: Type[ArraySchema]):
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
