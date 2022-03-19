"""Unit tests for pydantic datatype."""

import pandas as pd
import pytest
from pydantic import BaseModel

import pandera as pa
from pandera.engines.pandas_engine import PydanticModel


class Record(BaseModel):
    """Pydantic record model."""

    name: str
    xcoord: int
    ycoord: int


class PydanticSchema(pa.SchemaModel):
    """Pandera schema using the pydantic model."""

    class Config:
        """Config with dataframe-level data type."""

        dtype = PydanticModel(Record)
        coerce = True


class PanderaSchema(pa.SchemaModel):
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


def test_pydantic_model_init_errors():
    """SchemaInitError should be raised when coerce=False"""
    with pytest.raises(pa.errors.SchemaInitError):
        pa.DataFrameSchema(dtype=PydanticModel(Record), coerce=False)

    with pytest.raises(pa.errors.SchemaInitError):
        pa.SeriesSchema(dtype=PydanticModel(Record))

    with pytest.raises(pa.errors.SchemaInitError):
        pa.Column(dtype=PydanticModel(Record))

    with pytest.raises(pa.errors.SchemaInitError):
        pa.Index(dtype=PydanticModel(Record))
