"""Unit tests for Ibis table model."""

import pytest

from pandera.ibis import Column, DataFrameModel, DataFrameSchema


@pytest.fixture
def t_model_basic():
    class BasicModel(DataFrameModel):
        # string_col: str
        int_col: int

    return BasicModel


@pytest.fixture
def t_schema_basic():
    return DataFrameSchema(
        {
            # "string_col": Column(str),
            "int_col": Column(int),
        }
    )


def test_model_schema_equivalency(
    t_model_basic: DataFrameModel,
    t_schema_basic: DataFrameSchema,
):
    """Test that Ibis DataFrameModel and DataFrameSchema are equivalent."""
    t_schema_basic.name = "BasicModel"
    assert t_model_basic.to_schema() == t_schema_basic
