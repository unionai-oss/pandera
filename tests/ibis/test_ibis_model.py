"""Unit tests for Ibis table model."""

from typing import Annotated, Optional

import ibis
import ibis.expr.datatypes as dt
import pytest

import pandera.ibis as pa
from pandera.errors import SchemaError, SchemaInitError
from pandera.ibis import Column, DataFrameModel, DataFrameSchema
from pandera.typing import Column as TypingColumn


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
        string_col: str | None
        int_col: int

    schema = DataFrameSchema(
        name="ModelWithOptional",
        columns={
            "string_col": Column(dt.String, required=False),
            "int_col": Column(dt.Int64),
        },
    )
    assert ModelWithOptional.to_schema() == schema


def test_typing_column_runtime_contract():
    """Test the backend-neutral ``Column`` annotation with Ibis."""

    class Model(DataFrameModel):
        required: TypingColumn[dt.Int64]
        nullable_values: TypingColumn[dt.Int64 | None]
        optional_presence: TypingColumn[dt.Int64] | None
        explicit_nullable: TypingColumn[dt.Int64 | None] = pa.Field(
            nullable=False
        )

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


def test_typing_column_unsupported_dtype_error():
    """Unsupported ``Column`` dtypes receive a public schema error."""

    class UnsupportedDtype:
        pass

    class Invalid(DataFrameModel):
        bad: TypingColumn[UnsupportedDtype]

    with pytest.raises(SchemaInitError, match="Column dtype is not supported"):
        Invalid.to_schema()


def test_unsupported_dtype_error_is_reraised():
    class UnsupportedDtype:
        pass

    class Invalid(DataFrameModel):
        bad: UnsupportedDtype

    with pytest.raises((TypeError, ValueError)):
        Invalid.to_schema()


def test_annotated_field_metadata_propagation():
    """``Annotated[T, pa.Field(...)]`` should propagate the embedded
    ``FieldInfo`` metadata (description, title, unique, checks, etc.) to
    the ibis schema. See
    https://github.com/unionai-oss/pandera/issues/2110.
    """

    class Schema(DataFrameModel):
        name: Annotated[str, pa.Field(description="Name of the person")]
        age: int = pa.Field(ge=0, description="Age of the person")
        val: Annotated[float, pa.Field(ge=0.0, description="A value")]
        identifier: Annotated[int, pa.Field(unique=True, title="Identifier")]
        tag: Annotated[str, pa.Field(metadata={"k": "v"})]

    schema = Schema.to_schema()

    assert schema.columns["name"].description == "Name of the person"
    assert schema.columns["age"].description == "Age of the person"
    assert schema.columns["val"].description == "A value"
    assert schema.columns["identifier"].unique is True
    assert schema.columns["identifier"].title == "Identifier"
    assert schema.columns["tag"].metadata == {"k": "v"}

    # ``ge`` check defined inside the Annotated FieldInfo should also
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


def test_annotated_field_no_metadata_dedup():
    """Two ``Annotated`` annotations using independent ``pa.Field(...)``
    calls must not be deduplicated by Python's ``typing.Annotated`` cache.
    Without unique hashing on un-named ``FieldInfo`` instances, the second
    model would inadvertently inherit the first model's field configuration.
    """

    class ModelA(DataFrameModel):
        value: Annotated[int, pa.Field(ge=18, le=100)]

    class ModelB(DataFrameModel):
        value: Annotated[int, pa.Field(unique=True, title="ID")]

    schema_a = ModelA.to_schema()
    schema_b = ModelB.to_schema()

    assert len(schema_a.columns["value"].checks) == 2
    assert schema_b.columns["value"].unique is True
    assert schema_b.columns["value"].title == "ID"
    # ModelB should not have inherited ModelA's range checks.
    assert schema_b.columns["value"].checks == []
