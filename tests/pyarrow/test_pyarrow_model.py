"""Tests for the pyarrow DataFrameModel API."""

import pyarrow
import pytest

import pandera.pyarrow as pa
from pandera.errors import SchemaError
from pandera.typing.pyarrow import Table


class SimpleModel(pa.DataFrameModel):
    int_col: int
    str_col: str


def test_model_to_schema():
    schema = SimpleModel.to_schema()
    assert isinstance(schema, pa.DataFrameSchema)
    assert list(schema.columns) == ["int_col", "str_col"]


def test_model_validate():
    tbl = pyarrow.table({"int_col": [1, 2], "str_col": ["a", "b"]})
    assert SimpleModel.validate(tbl).equals(tbl)


def test_model_validate_wrong_dtype():
    tbl = pyarrow.table({"int_col": ["x"], "str_col": ["a"]})
    with pytest.raises(SchemaError):
        SimpleModel.validate(tbl)


def test_model_with_field_checks():
    class Bounded(pa.DataFrameModel):
        a: int = pa.Field(gt=0, le=10)

    assert Bounded.validate(pyarrow.table({"a": [1, 10]})) is not None
    with pytest.raises(SchemaError):
        Bounded.validate(pyarrow.table({"a": [0]}))
    with pytest.raises(SchemaError):
        Bounded.validate(pyarrow.table({"a": [11]}))


def test_model_optional_column():
    class WithOptional(pa.DataFrameModel):
        a: int
        b: str | None

    # 'b' is optional, so a table without it validates...
    tbl = pyarrow.table({"a": [1]})
    assert WithOptional.validate(tbl).equals(tbl)

    # ...but a required column that is missing still fails.
    class AllRequired(pa.DataFrameModel):
        a: int
        b: str

    with pytest.raises(SchemaError):
        AllRequired.validate(tbl)


def test_model_nullable_field():
    class Nullable(pa.DataFrameModel):
        a: int = pa.Field(nullable=True)

    tbl = pyarrow.table({"a": [1, None]})
    assert Nullable.validate(tbl).equals(tbl)


def test_model_custom_check():
    import pyarrow.compute as pc

    class WithCheck(pa.DataFrameModel):
        a: int

        @pa.check("a")
        @classmethod
        def a_is_positive(cls, data):
            return pc.greater(data.table[data.key], 0)

    assert WithCheck.validate(pyarrow.table({"a": [1, 2]})) is not None
    with pytest.raises(SchemaError):
        WithCheck.validate(pyarrow.table({"a": [-1]}))


def test_model_empty():
    empty = SimpleModel.empty()
    assert empty.num_rows == 0
    assert empty.column_names == ["int_col", "str_col"]
    assert empty.schema.field("int_col").type == pyarrow.int64()


def test_check_types_decorator():
    @pa.check_types
    def transform(tbl: Table[SimpleModel]) -> Table[SimpleModel]:
        return tbl

    valid = pyarrow.table({"int_col": [1], "str_col": ["a"]})
    assert transform(valid).equals(valid)

    invalid = pyarrow.table({"int_col": ["x"], "str_col": ["a"]})
    with pytest.raises(SchemaError):
        transform(invalid)
