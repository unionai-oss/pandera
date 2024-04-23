"""Unit tests for polars dataframe model."""

import sys
from typing import Optional

import pytest

import polars as pl
from pandera.errors import SchemaError
from pandera.polars import (
    DataFrameModel,
    DataFrameSchema,
    Column,
    PolarsData,
    Field,
    check,
    dataframe_check,
)


@pytest.fixture
def ldf_model_basic():
    class BasicModel(DataFrameModel):
        string_col: str
        int_col: int

    return BasicModel


@pytest.fixture
def ldf_schema_basic():
    return DataFrameSchema(
        {
            "string_col": Column(pl.Utf8),
            "int_col": Column(pl.Int64),
        },
    )


@pytest.fixture
def ldf_model_with_fields():
    class ModelWithFields(DataFrameModel):
        string_col: str = Field(isin=[*"abc"])
        int_col: int = Field(ge=0)

    return ModelWithFields


@pytest.fixture
def ldf_model_with_custom_column_checks():
    class ModelWithCustomColumnChecks(DataFrameModel):
        string_col: str
        int_col: int

        @check("string_col")
        @classmethod
        def custom_isin(cls, data: PolarsData) -> pl.LazyFrame:
            return data.lazyframe.select(pl.col(data.key).is_in([*"abc"]))

        @check("int_col")
        @classmethod
        def custom_ge(cls, data: PolarsData) -> pl.LazyFrame:
            return data.lazyframe.select(pl.col(data.key).ge(0))

    return ModelWithCustomColumnChecks


@pytest.fixture
def ldf_model_with_custom_dataframe_checks():
    class ModelWithCustomDataFrameChecks(DataFrameModel):
        string_col: str
        int_col: int

        @dataframe_check
        @classmethod
        def not_empty(cls, data: PolarsData) -> pl.LazyFrame:
            return data.lazyframe.select(pl.count().gt(0))

    return ModelWithCustomDataFrameChecks


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


def test_model_schema_equivalency_with_optional():
    class ModelWithOptional(DataFrameModel):
        string_col: Optional[str]
        int_col: int

    schema = DataFrameSchema(
        name="ModelWithOptional",
        columns={
            "string_col": Column(pl.Utf8, required=False),
            "int_col": Column(pl.Int64),
        },
    )
    assert ModelWithOptional.to_schema() == schema


@pytest.mark.parametrize(
    "column_mod,exception_cls",
    [
        # this modification will cause a ComputeError since casting the values
        # in ldf_basic will cause the error outside of pandera validation
        ({"string_col": pl.Int64}, pl.exceptions.ComputeError),
        # this modification will cause a SchemaError since schema validation
        # can actually catch the type mismatch
        ({"int_col": pl.Utf8}, SchemaError),
        ({"int_col": pl.Float64}, SchemaError),
    ],
)
def test_basic_model(
    column_mod,
    exception_cls,
    ldf_model_basic: DataFrameModel,
    ldf_basic: pl.LazyFrame,
):
    """Test basic polars lazy dataframe."""
    query = ldf_basic.pipe(ldf_model_basic.validate)
    df = query.collect()
    assert isinstance(query, pl.LazyFrame)
    assert isinstance(df, pl.DataFrame)

    invalid_df = ldf_basic.cast(column_mod)

    with pytest.raises(exception_cls):
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


def test_model_with_custom_column_checks(
    ldf_model_with_custom_column_checks,
    ldf_basic,
):
    query = ldf_basic.pipe(ldf_model_with_custom_column_checks.validate)
    df = query.collect()
    assert isinstance(query, pl.LazyFrame)
    assert isinstance(df, pl.DataFrame)

    invalid_df = ldf_basic.with_columns(
        string_col=pl.lit("x"), int_col=pl.lit(-1)
    )
    with pytest.raises(SchemaError):
        invalid_df.pipe(ldf_model_with_custom_column_checks.validate).collect()


def test_model_with_custom_dataframe_checks(
    ldf_model_with_custom_dataframe_checks,
    ldf_basic,
):
    query = ldf_basic.pipe(ldf_model_with_custom_dataframe_checks.validate)
    df = query.collect()
    assert isinstance(query, pl.LazyFrame)
    assert isinstance(df, pl.DataFrame)

    # remove all rows
    invalid_df = ldf_basic.filter(pl.lit(False))
    with pytest.raises(SchemaError):
        invalid_df.pipe(
            ldf_model_with_custom_dataframe_checks.validate
        ).collect()


@pytest.fixture
def schema_with_list_type():
    return DataFrameSchema(
        name="ModelWithNestedDtypes",
        columns={
            "list_col": Column(pl.List(pl.Utf8)),
        },
    )


@pytest.mark.skipif(
    sys.version_info < (3, 9),
    reason="standard collection generics are not supported in python < 3.9",
)
def test_polars_python_list_df_model(schema_with_list_type):
    class ModelWithNestedDtypes(DataFrameModel):
        # pylint: disable=unsubscriptable-object
        list_col: list[str]

    schema = ModelWithNestedDtypes.to_schema()
    assert schema_with_list_type == schema
