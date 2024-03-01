"""Unit tests for polars dataframe model."""

import pytest

import polars as pl
from pandera.errors import SchemaError
from pandera.polars import DataFrameModel, DataFrameSchema, Column, Field


@pytest.fixture
def ldf_model_basic():
    class BasicModel(DataFrameModel):
        string_col: str
        int_col: int

    return BasicModel


@pytest.fixture
def ldf_model_with_fields():
    class ModelWithFields(DataFrameModel):
        string_col: str = Field(isin=[*"abc"])
        int_col: int = Field(ge=0)

    return ModelWithFields


@pytest.fixture
def ldf_schema_basic():
    return DataFrameSchema(
        {
            "string_col": Column(pl.Utf8),
            "int_col": Column(pl.Int64),
        },
    )


@pytest.fixture
def ldf_basic():
    """Basic polars lazy dataframe fixture."""
    return pl.DataFrame(
        {
            "string_col": ["a", "b", "c"],
            "int_col": [0, 1, 2],
        }
    ).lazy()


def test_model_schema_equivalency(
    ldf_model_basic: DataFrameModel,
    ldf_schema_basic: DataFrameSchema,
):
    """Test that polars DataFrameModel and DataFrameSchema are equivalent."""
    ldf_schema_basic.name = "BasicModel"
    assert ldf_model_basic.to_schema() == ldf_schema_basic


def test_basic_model(ldf_model_basic: DataFrameModel, ldf_basic: pl.LazyFrame):
    """Test basic polars lazy dataframe."""
    query = ldf_basic.pipe(ldf_model_basic.validate)
    df = query.collect()
    assert isinstance(query, pl.LazyFrame)
    assert isinstance(df, pl.DataFrame)

    invalid_string_type_df = ldf_basic.cast({"string_col": pl.Int64})
    invalid_int_type_df = ldf_basic.cast({"int_col": pl.Utf8})

    for invalid_df in (invalid_string_type_df, invalid_int_type_df):
        with pytest.raises(SchemaError):
            invalid_df.pipe(ldf_model_basic.validate).collect()


def test_model_with_fields(ldf_model_with_fields, ldf_basic):
    query = ldf_basic.pipe(ldf_model_with_fields.validate)
    df = query.collect()
    assert isinstance(query, pl.LazyFrame)
    assert isinstance(df, pl.DataFrame)

    invalid_df = ldf_basic.with_columns(
        string_col=pl.lit("x"), int_col=pl.lit(-1)
    )
    with pytest.raises(SchemaError):
        invalid_df.pipe(ldf_model_with_fields.validate).collect()
