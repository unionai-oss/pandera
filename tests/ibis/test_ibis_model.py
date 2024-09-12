"""Unit tests for Ibis table model."""

from typing import Optional

import pytest
import ibis.expr.datatypes as dt

from pandera.ibis import Column, DataFrameModel, DataFrameSchema


@pytest.fixture
def t_model_basic():
    class BasicModel(DataFrameModel):
        string_col: str
        int_col: int

    return BasicModel


@pytest.fixture
def t_schema_basic():
    return DataFrameSchema(
        {
            "string_col": Column(dt.String),
            "int_col": Column(dt.Int64),
        }
    )


def test_model_schema_equivalency(
    t_model_basic: DataFrameModel,
    t_schema_basic: DataFrameSchema,
):
    """Test that Ibis DataFrameModel and DataFrameSchema are equivalent."""
    t_schema_basic.name = "BasicModel"
    assert t_model_basic.to_schema() == t_schema_basic


def test_model_schema_equivalency_with_optional():
    class ModelWithOptional(DataFrameModel):
        string_col: Optional[str]
        int_col: int

    schema = DataFrameSchema(
        name="ModelWithOptional",
        columns={
            "string_col": Column(dt.String, required=False),
            "int_col": Column(dt.Int64),
        },
    )
    assert ModelWithOptional.to_schema() == schema
