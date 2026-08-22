"""Unit tests for nested DataFrameModel/DataFrameSchema support (polars).

See https://github.com/unionai-oss/pandera/issues/2425.
"""

import polars as pl
import pytest

import pandera.polars as pa
from pandera.engines.polars_engine import PanderaSchema


class Foo(pa.DataFrameModel):
    """Simple nested model with a non-nullable field."""

    x: int = pa.Field(nullable=False)


class Bar(pa.DataFrameModel):
    """Model with a nested `Foo` column."""

    foo: Foo


# ---------------------------------------------------------------------------
# Basic behavior, matching the examples in issue #2425
# ---------------------------------------------------------------------------


def test_nested_model_dtype_is_struct():
    """The nested column's dtype should be the struct derived from Foo."""
    schema = Bar.to_schema()
    column = schema.columns["foo"]
    assert isinstance(column.dtype, PanderaSchema)
    assert column.dtype.type == pl.Struct({"x": pl.Int64})
    assert column.coerce, "nested columns should auto-coerce"


def test_nested_model_wrong_inner_dtype():
    data = pl.DataFrame({"foo": [{"x": "not an int"}]})
    with pytest.raises(pa.errors.SchemaError):
        Bar.validate(data)


def test_nested_model_nullability_violation():
    data = pl.DataFrame({"foo": [{"x": 1}, {"x": None}]})
    with pytest.raises(pa.errors.SchemaError):
        Bar.validate(data)


def test_nested_model_valid_data():
    data = pl.DataFrame({"foo": [{"x": 1}, {"x": 2}]})
    validated = Bar.validate(data)
    assert isinstance(validated, pl.DataFrame)
    assert validated["foo"].struct.field("x").to_list() == [1, 2]


# ---------------------------------------------------------------------------
# Checks (not just dtype/nullability) on nested fields
# ---------------------------------------------------------------------------


class Positive(pa.DataFrameModel):
    y: int = pa.Field(gt=0)


class HasPositive(pa.DataFrameModel):
    positive: Positive


def test_nested_model_check_violation():
    data = pl.DataFrame({"positive": [{"y": 5}, {"y": -1}]})
    with pytest.raises(pa.errors.SchemaError):
        HasPositive.validate(data)


def test_nested_model_check_passes():
    data = pl.DataFrame({"positive": [{"y": 5}, {"y": 1}]})
    validated = HasPositive.validate(data)
    assert validated["positive"].struct.field("y").to_list() == [5, 1]


# ---------------------------------------------------------------------------
# Multiple levels of nesting
# ---------------------------------------------------------------------------


class Inner(pa.DataFrameModel):
    y: int = pa.Field(gt=0)


class Middle(pa.DataFrameModel):
    x: int = pa.Field(nullable=False)
    inner: Inner


class Outer(pa.DataFrameModel):
    middle: Middle


def test_deeply_nested_model_invalid():
    data = pl.DataFrame(
        {
            "middle": [
                {"x": 1, "inner": {"y": 5}},
                {"x": 2, "inner": {"y": -1}},
            ]
        }
    )
    with pytest.raises(pa.errors.SchemaError):
        Outer.validate(data)


def test_deeply_nested_model_valid():
    data = pl.DataFrame(
        {
            "middle": [
                {"x": 1, "inner": {"y": 5}},
                {"x": 2, "inner": {"y": 7}},
            ]
        }
    )
    validated = Outer.validate(data)
    assert isinstance(validated, pl.DataFrame)


# ---------------------------------------------------------------------------
# lazy=True error collection
# ---------------------------------------------------------------------------


def test_nested_model_lazy_collects_failure_cases():
    data = pl.DataFrame({"foo": [{"x": 1}, {"x": None}, {"x": 3}]})
    with pytest.raises(pa.errors.SchemaErrors) as excinfo:
        Bar.validate(data, lazy=True)
    failure_cases = excinfo.value.failure_cases
    assert failure_cases.shape[0] >= 1


# ---------------------------------------------------------------------------
# A null *struct* (missing nested record) is governed by the outer column's
# own nullability, not by the nested schema's field-level constraints.
# ---------------------------------------------------------------------------


class OptionalFooModel(pa.DataFrameModel):
    foo: Foo = pa.Field(nullable=True)


def test_nullable_outer_struct_allows_null_record():
    data = pl.DataFrame(
        {"foo": [{"x": 1}, None]},
        schema={"foo": pl.Struct({"x": pl.Int64})},
    )
    validated = OptionalFooModel.validate(data)
    assert validated["foo"].to_list() == [{"x": 1}, None]


def test_non_nullable_outer_struct_rejects_null_record():
    data = pl.DataFrame(
        {"foo": [{"x": 1}, None]},
        schema={"foo": pl.Struct({"x": pl.Int64})},
    )
    with pytest.raises(pa.errors.SchemaError, match="non-nullable"):
        Bar.validate(data)


# ---------------------------------------------------------------------------
# Works with the object-based DataFrameSchema API too, per the maintainer's
# request that this be implemented at the type-engine level for both APIs.
# ---------------------------------------------------------------------------


def test_nested_dataframe_schema_api():
    foo_schema = pa.DataFrameSchema({"x": pa.Column(int, nullable=False)})
    bar_schema = pa.DataFrameSchema(
        {"foo": pa.Column(foo_schema, coerce=True)}
    )

    valid = pl.DataFrame({"foo": [{"x": 1}, {"x": 2}]})
    assert bar_schema.validate(valid) is not None

    invalid = pl.DataFrame({"foo": [{"x": None}]})
    with pytest.raises(pa.errors.SchemaError):
        bar_schema.validate(invalid)


def test_panderaschema_dtype_directly():
    """PanderaSchema can also be used directly as a Column dtype."""
    foo_schema = pa.DataFrameSchema({"x": pa.Column(int, nullable=False)})
    bar_schema = pa.DataFrameSchema(
        {"foo": pa.Column(PanderaSchema(foo_schema))}
    )
    assert bar_schema.columns["foo"].coerce

    invalid = pl.DataFrame({"foo": [{"x": None}]})
    with pytest.raises(pa.errors.SchemaError):
        bar_schema.validate(invalid)


def test_panderaschema_rejects_non_schema_input():
    with pytest.raises(TypeError):
        PanderaSchema(42)


# ---------------------------------------------------------------------------
# LazyFrame input
# ---------------------------------------------------------------------------


def test_nested_model_with_lazyframe():
    lf = pl.LazyFrame({"foo": [{"x": 1}, {"x": 2}]})
    validated = Bar.validate(lf)
    assert isinstance(validated, pl.LazyFrame)
    assert validated.collect()["foo"].struct.field("x").to_list() == [1, 2]


# ---------------------------------------------------------------------------
# List[NestedModel] isn't supported yet -- must fail loudly, not silently
# mis-validate by dropping the list wrapper.
# ---------------------------------------------------------------------------


def test_list_of_nested_model_raises_clear_error():
    with pytest.raises(pa.errors.SchemaInitError, match="not yet supported"):

        class HasListOfFoo(pa.DataFrameModel):
            foos: list[Foo]

        HasListOfFoo.to_schema()