"""Tests for the DataFusion DataFrameModel API."""

import datetime

import datafusion
import pyarrow
from datafusion import col, lit

import pandera.datafusion as pa
from pandera.errors import SchemaErrorReason
from pandera.typing import FieldType


class SimpleModel(pa.DataFrameModel):
    int_col: int
    str_col: str


# Building the schema


def test_model_to_schema():
    schema = SimpleModel.to_schema()
    assert isinstance(schema, pa.DataFrameSchema)
    assert list(schema.columns) == ["int_col", "str_col"]
    assert all(isinstance(c, pa.Column) for c in schema.columns.values())


def test_model_schema_equivalency():
    class Model(pa.DataFrameModel):
        a: int = pa.Field(gt=0)
        b: str = pa.Field(nullable=True)
        c: float | None

    schema = pa.DataFrameSchema(
        {
            "a": pa.Column(int, pa.Check.gt(0)),
            "b": pa.Column(str, nullable=True),
            "c": pa.Column(float, required=False),
        },
        name="Model",
    )
    assert Model.to_schema() == schema


def test_model_field_type_presence_and_nullability():
    class Model(pa.DataFrameModel):
        required: int
        nullable_values: FieldType[int | None]
        optional_presence: FieldType[int] | None
        explicit_nullable: int = pa.Field(nullable=True)

    schema = Model.to_schema()
    assert schema.columns["required"].dtype == pa.Column(int).dtype
    assert schema.columns["required"].required
    assert not schema.columns["required"].nullable
    assert schema.columns["nullable_values"].required
    assert schema.columns["nullable_values"].nullable
    assert not schema.columns["optional_presence"].required
    assert schema.columns["explicit_nullable"].nullable


def test_model_field_type_contract(make_df, fails):
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

    valid = {"renamed": [1], "nullable": [1], "assigned": [2]}
    assert Model.validate(make_df(valid)) is not None
    fails(
        Model,
        make_df({**valid, "renamed": [0]}),
        SchemaErrorReason.DATAFRAME_CHECK,
    )


def test_model_field_access_returns_column_name():
    class Model(pa.DataFrameModel):
        a: int
        b: str = pa.Field(alias="renamed")

    assert Model.a == "a"
    assert Model.b == "renamed"


def test_model_inheritance():
    class Base(pa.DataFrameModel):
        a: int = pa.Field(gt=0)

        class Config:
            strict = True

    class Child(Base):
        b: str

    schema = Child.to_schema()
    assert list(schema.columns) == ["a", "b"]
    assert schema.columns["a"].checks
    assert schema.strict


def test_model_config(make_df, fails):
    class Model(pa.DataFrameModel):
        a: int

        class Config:
            name = "Named"
            strict = True
            ordered = True

    schema = Model.to_schema()
    assert schema.name == "Named"
    assert schema.strict
    assert schema.ordered
    fails(
        Model,
        make_df({"a": [1], "extra": [1]}),
        SchemaErrorReason.COLUMN_NOT_IN_SCHEMA,
    )


def test_model_pyarrow_dtype_annotation(make_df, fails):
    class ArrowTyped(pa.DataFrameModel):
        ts: pyarrow.timestamp("us")  # type: ignore[valid-type]

    table = pyarrow.table(
        {"ts": pyarrow.array([datetime.datetime(2024, 1, 1)])}
    )
    assert ArrowTyped.validate(make_df(table)) is not None
    fails(ArrowTyped, make_df({"ts": [1]}), SchemaErrorReason.WRONG_DATATYPE)


# Validation


def test_model_validate(make_df):
    data = {"int_col": [1, 2], "str_col": ["a", "b"]}
    validated = SimpleModel.validate(make_df(data))
    assert isinstance(validated, datafusion.DataFrame)
    assert validated.to_pydict() == data


def test_model_validate_wrong_dtype(make_df, fails):
    fails(
        SimpleModel,
        make_df({"int_col": ["x"], "str_col": ["a"]}),
        SchemaErrorReason.WRONG_DATATYPE,
    )


def test_model_validate_lazy(make_df, fails):
    class Model(pa.DataFrameModel):
        a: int = pa.Field(gt=0)
        b: str

    error = fails(
        Model,
        make_df({"a": [-1], "b": [1]}),
        SchemaErrorReason.DATAFRAME_CHECK,
        SchemaErrorReason.WRONG_DATATYPE,
        lazy=True,
    )
    assert isinstance(error.failure_cases, pyarrow.Table)


def test_model_with_field_checks(make_df, fails):
    class Bounded(pa.DataFrameModel):
        a: int = pa.Field(gt=0, le=10)

    assert Bounded.validate(make_df({"a": [1, 10]})) is not None
    for invalid in ([0], [11]):
        fails(
            Bounded,
            make_df({"a": invalid}),
            SchemaErrorReason.DATAFRAME_CHECK,
        )


def test_model_optional_column(make_df, fails):
    class WithOptional(pa.DataFrameModel):
        a: int
        b: str | None

    class AllRequired(pa.DataFrameModel):
        a: int
        b: str

    df = make_df({"a": [1]})
    assert WithOptional.validate(df).to_pydict() == {"a": [1]}
    fails(AllRequired, df, SchemaErrorReason.COLUMN_NOT_IN_DATAFRAME)


def test_model_nullable_field(make_df, fails):
    class Nullable(pa.DataFrameModel):
        a: int = pa.Field(nullable=True)

    class NotNullable(pa.DataFrameModel):
        a: int

    df = make_df({"a": [1, None]})
    assert Nullable.validate(df).to_pydict() == {"a": [1, None]}
    fails(NotNullable, df, SchemaErrorReason.SERIES_CONTAINS_NULLS)


def test_model_unique_field(make_df, fails):
    class Unique(pa.DataFrameModel):
        a: int = pa.Field(unique=True)

    assert Unique.validate(make_df({"a": [1, 2]})) is not None
    fails(
        Unique,
        make_df({"a": [1, 1]}),
        SchemaErrorReason.SERIES_CONTAINS_DUPLICATES,
    )


def test_model_custom_check(make_df, fails):
    class WithCheck(pa.DataFrameModel):
        a: int

        @pa.check("a")
        @classmethod
        def a_is_positive(cls, data):
            return col(data.key) > lit(0)

    assert WithCheck.validate(make_df({"a": [1, 2]})) is not None
    fails(WithCheck, make_df({"a": [-1]}), SchemaErrorReason.DATAFRAME_CHECK)


def test_model_dataframe_check(make_df, fails):
    class WithFrameCheck(pa.DataFrameModel):
        a: int
        b: int

        @pa.dataframe_check
        @classmethod
        def a_lt_b(cls, data):
            return col("a") < col("b")

    assert WithFrameCheck.validate(make_df({"a": [1], "b": [2]})) is not None
    fails(
        WithFrameCheck,
        make_df({"a": [3], "b": [2]}),
        SchemaErrorReason.DATAFRAME_CHECK,
    )


# Empty frames


def test_model_empty():
    empty = SimpleModel.empty()
    assert isinstance(empty, datafusion.DataFrame)
    assert empty.count() == 0
    assert list(empty.schema().names) == ["int_col", "str_col"]
    assert empty.schema().field("int_col").type == pyarrow.int64()
    assert SimpleModel.validate(empty) is not None


def test_model_empty_without_columns():
    class NoColumns(pa.DataFrameModel):
        pass

    assert list(NoColumns.empty().schema().names) == []
