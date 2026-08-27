"""Tests for the pyarrow DataFrameModel API."""

import pyarrow
import pytest

import pandera.pyarrow as pa
from pandera.errors import SchemaError
from pandera.typing import FieldType
from pandera.typing.pyarrow import Table


class SimpleModel(pa.DataFrameModel):
    int_col: int
    str_col: str


def test_model_to_schema():
    schema = SimpleModel.to_schema()
    assert isinstance(schema, pa.DataFrameSchema)
    assert list(schema.columns) == ["int_col", "str_col"]


def test_model_field_type_presence_and_nullability():
    """Test the ``FieldType`` presence and nullability contract."""

    class Model(pa.DataFrameModel):
        required: int
        nullable_values: int | None
        optional_presence: FieldType[int] | None
        explicit_nullable: int | None = pa.Field(nullable=False)

    schema = Model.to_schema()
    assert schema.columns["required"].dtype == pa.Column(int).dtype
    assert schema.columns["required"].required
    assert not schema.columns["required"].nullable
    assert schema.columns["nullable_values"].required
    assert schema.columns["nullable_values"].nullable
    assert not schema.columns["optional_presence"].required
    assert not schema.columns["explicit_nullable"].nullable
    assert Model.required == "required"
    assert isinstance(Model.optional_presence, str)


def test_field_type_contract():
    """The typing-only field marker composes with runtime field metadata."""

    class Model(pa.DataFrameModel):
        checked: FieldType[
            int,
            pa.Field(
                alias="renamed",
                description="checked field",
                metadata={"source": "typing-field"},
                title="Checked",
                unique=True,
                gt=0,
            ),
        ]
        nullable: FieldType[int | None, pa.Field()]
        optional: FieldType[str] | None
        optional_metadata: FieldType[str, pa.Field(required=False)]
        assigned: FieldType[int] = pa.Field(description="assigned")

    schema = Model.to_schema()
    checked = schema.columns["renamed"]
    assert checked.required
    assert not checked.nullable
    assert checked.description == "checked field"
    assert checked.metadata == {"source": "typing-field"}
    assert checked.title == "Checked"
    assert checked.unique
    assert checked.checks
    assert schema.columns["nullable"].required
    assert schema.columns["nullable"].nullable
    assert not schema.columns["optional"].required
    assert not schema.columns["optional_metadata"].required
    assert schema.columns["assigned"].description == "assigned"
    assert Model.checked == "renamed"

    valid = pyarrow.table(
        {
            "renamed": [1],
            "nullable": pyarrow.array([1], type=pyarrow.int64()),
            "assigned": [2],
        }
    )
    Model.validate(valid)
    with pytest.raises(SchemaError):
        Model.validate(valid.set_column(0, "renamed", pyarrow.array([0])))


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
        b: FieldType[str] | None

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
