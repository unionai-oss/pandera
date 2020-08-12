"""Tests schema creation and validation from type annotations."""

import pandas as pd
import pytest

import pandera as pa
from pandera import Column, DataFrameSchema, check_types
from pandera.errors import SchemaError
from pandera.typing import DataFrame, Index, SchemaModel, Series


def test_schemamodel_to_dataframeschema():
    """Tests that a SchemaModel.get_schema() can produce the correct schema."""

    class Schema(SchemaModel):
        a: Series["int"]
        b: Series["string"]
        idx: Index["string"]

    expected = DataFrameSchema(
        columns={"a": Column("int"), "b": Column("string")}, index=pa.Index("string")
    )

    assert expected == Schema.get_schema()


def test_check_types():
    class A(SchemaModel):
        a: Series["int"]

    @check_types()
    def transform(df: DataFrame[A]) -> DataFrame[A]:
        return df

    df = pd.DataFrame({"a": [1]})
    try:
        transform(df)
    except Exception as e:
        pytest.fail(f"Unexpected Exception {e}")


def test_check_types_errors():
    class A(SchemaModel):
        a: Series["int"]
        idx: Index["string"]

    class B(SchemaModel):
        b: Series["int"]

    df = pd.DataFrame({"a": [1]}, index=["1"])

    @check_types()
    def transform_index(df) -> DataFrame[A]:
        return df.reset_index(drop=True)

    with pytest.raises(SchemaError):
        transform_index(df)

    @check_types()
    def to_b(df: DataFrame[A]) -> DataFrame[B]:
        return df

    with pytest.raises(SchemaError, match="column 'b' not in dataframe"):
        to_b(df)

    @check_types()
    def to_str(df: DataFrame[A]) -> DataFrame[A]:
        df["a"] = "1"
        return df

    with pytest.raises(SchemaError, match="expected series 'a' to have type int64"):
        to_str(df)
