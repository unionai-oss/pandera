"""Unit tests for Ibis table model."""

from typing import Optional

import ibis
import ibis.expr.datatypes as dt
import pytest

import pandera.ibis as pa
from pandera.errors import SchemaError
from pandera.ibis import Column, DataFrameModel, DataFrameSchema
from pandera.typing import FieldType


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


def test_model_schema_equivalency_with_nullable():
    class ModelWithNullable(DataFrameModel):
        string_col: str | None
        int_col: int

    schema = DataFrameSchema(
        name="ModelWithNullable",
        columns={
            "string_col": Column(dt.String, nullable=True),
            "int_col": Column(dt.Int64),
        },
    )
    assert ModelWithNullable.to_schema() == schema


def test_field_type_presence_and_nullability():
    """Test the ``FieldType`` presence and nullability contract."""

    class Model(DataFrameModel):
        required: dt.Int64
        nullable_values: dt.Int64 | None
        optional_presence: FieldType[dt.Int64] | None
        explicit_nullable: dt.Int64 | None = pa.Field(nullable=False)

    schema = Model.to_schema()
    assert schema.columns["required"].dtype == Column(dt.Int64).dtype
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

    class Model(DataFrameModel):
        checked: FieldType[
            dt.Int64,
            pa.Field(
                alias="renamed",
                description="checked field",
                metadata={"source": "typing-field"},
                title="Checked",
                unique=True,
                gt=0,
            ),
        ]
        nullable: FieldType[dt.Int64 | None, pa.Field()]
        optional: FieldType[dt.String] | None
        optional_metadata: FieldType[dt.String, pa.Field(required=False)]
        assigned: FieldType[dt.Int64] = pa.Field(description="assigned")

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

    valid = ibis.memtable({"renamed": [1], "nullable": [1], "assigned": [2]})
    Model.validate(valid)
    with pytest.raises(SchemaError):
        Model.validate(
            ibis.memtable({"renamed": [0], "nullable": [1], "assigned": [2]})
        )


def test_field_type_metadata_propagation():
    """``FieldType[T, pa.Field(...)]`` should propagate field metadata
    (description, title, unique, checks, etc.) to
    the ibis schema. See
    https://github.com/unionai-oss/pandera/issues/2110.
    """

    class Schema(DataFrameModel):
        name: FieldType[str, pa.Field(description="Name of the person")]
        age: int = pa.Field(ge=0, description="Age of the person")
        val: FieldType[float, pa.Field(ge=0.0, description="A value")]
        identifier: FieldType[int, pa.Field(unique=True, title="Identifier")]
        tag: FieldType[str, pa.Field(metadata={"k": "v"})]

    schema = Schema.to_schema()

    assert schema.columns["name"].description == "Name of the person"
    assert schema.columns["age"].description == "Age of the person"
    assert schema.columns["val"].description == "A value"
    assert schema.columns["identifier"].unique is True
    assert schema.columns["identifier"].title == "Identifier"
    assert schema.columns["tag"].metadata == {"k": "v"}

    # ``ge`` check defined inside the FieldType metadata should also
    # be applied during validation.
    valid = ibis.memtable(
        {
            "name": ["Alice"],
            "age": [25],
            "val": [1.0],
            "identifier": [1],
            "tag": ["x"],
        }
    )
    Schema.validate(valid)

    invalid = ibis.memtable(
        {
            "name": ["Alice"],
            "age": [25],
            "val": [-1.0],
            "identifier": [1],
            "tag": ["x"],
        }
    )
    with pytest.raises(SchemaError):
        Schema.validate(invalid)


def test_field_type_metadata_no_dedup():
    """Independent ``FieldType`` metadata objects must remain distinct."""

    class ModelA(DataFrameModel):
        value: FieldType[int, pa.Field(ge=18, le=100)]

    class ModelB(DataFrameModel):
        value: FieldType[int, pa.Field(unique=True, title="ID")]

    schema_a = ModelA.to_schema()
    schema_b = ModelB.to_schema()

    assert len(schema_a.columns["value"].checks) == 2
    assert schema_b.columns["value"].unique is True
    assert schema_b.columns["value"].title == "ID"
    # ModelB should not have inherited ModelA's range checks.
    assert schema_b.columns["value"].checks == []
