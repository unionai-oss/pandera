"""Tests for standalone DataFusion ``Column`` validation."""

import datafusion
import pytest

import pandera.datafusion as pa
from pandera.config import config_context
from pandera.errors import (
    SchemaDefinitionError,
    SchemaErrorReason,
    SchemaWarning,
)


def test_column_validate_returns_datafusion_dataframe(make_df):
    df = make_df({"a": [1, 2], "b": ["x", "y"]})
    validated = pa.Column(int, pa.Check.gt(0), name="a").validate(df)
    assert isinstance(validated, datafusion.DataFrame)
    assert validated.to_pydict() == {"a": [1, 2], "b": ["x", "y"]}


def test_column_without_name_raises(make_df):
    with pytest.raises(
        SchemaDefinitionError,
        match="Column schema must have a name specified",
    ):
        pa.Column(int).validate(make_df({"a": [1]}))


def test_column_set_name(make_df):
    column = pa.Column(int).set_name("a")
    assert column.name == "a"
    assert column.validate(make_df({"a": [1]})) is not None


def test_column_inplace_warns(make_df):
    with pytest.warns(UserWarning, match="inplace=True will have no effect"):
        pa.Column(int, name="a").validate(make_df({"a": [1]}), inplace=True)


def test_column_wrong_dtype(make_df, fails):
    fails(
        pa.Column(int, name="a"),
        make_df({"a": ["x"]}),
        SchemaErrorReason.WRONG_DATATYPE,
    )


def test_column_check(make_df, fails):
    error = fails(
        pa.Column(int, pa.Check.gt(0), name="a"),
        make_df({"a": [1, -2]}),
        SchemaErrorReason.DATAFRAME_CHECK,
    )
    assert error.failure_cases.column("a").to_pylist() == [-2]


def test_column_nullable(make_df, fails):
    df = make_df({"a": [1, None]})
    assert pa.Column(int, name="a", nullable=True).validate(df) is not None
    fails(
        pa.Column(int, name="a"), df, SchemaErrorReason.SERIES_CONTAINS_NULLS
    )


def test_column_unique(make_df, fails):
    column = pa.Column(int, name="a", unique=True)
    assert column.validate(make_df({"a": [1, 2]})) is not None
    fails(
        column,
        make_df({"a": [1, 1]}),
        SchemaErrorReason.SERIES_CONTAINS_DUPLICATES,
    )


@pytest.mark.parametrize(
    "column_kwargs",
    [
        {"name": r"^col_\d$"},
        {"name": r"col_\d", "regex": True},
    ],
)
def test_column_regex(make_df, fails, column_kwargs):
    column = pa.Column(int, **column_kwargs)
    assert column.regex
    df = make_df({"col_1": [1], "col_2": [2], "other": ["x"]})
    assert column.validate(df).to_pydict() == df.to_pydict()

    fails(
        column,
        make_df({"col_1": [1], "col_2": ["x"]}),
        SchemaErrorReason.WRONG_DATATYPE,
    )


def test_column_regex_without_match(make_df, fails):
    fails(
        pa.Column(int, name=r"^col_\d$"),
        make_df({"other": [1]}),
        SchemaErrorReason.INVALID_COLUMN_NAME,
    )


@pytest.mark.xfail(
    reason="the narwhals column backend raises a bare KeyError when the "
    "column is absent from the frame",
    raises=KeyError,
    strict=True,
)
def test_column_missing_from_dataframe(make_df, fails):
    fails(
        pa.Column(int, name="a"),
        make_df({"b": [1]}),
        SchemaErrorReason.COLUMN_NOT_IN_DATAFRAME,
    )


def test_column_lazy_collects_errors(make_df, fails):
    error = fails(
        pa.Column(int, pa.Check.gt(0), name="a"),
        make_df({"a": [1, -2, -3]}),
        SchemaErrorReason.DATAFRAME_CHECK,
        lazy=True,
    )
    assert error.failure_cases.column("failure_case").to_pylist() == [
        "-2",
        "-3",
    ]


def test_column_drop_invalid_rows(make_df):
    column = pa.Column(int, pa.Check.gt(0), name="a", drop_invalid_rows=True)
    validated = column.validate(make_df({"a": [1, -2, 3]}), lazy=True)
    assert validated.to_pydict() == {"a": [1, 3]}


@pytest.mark.usefixtures("default_depth")
def test_column_default_depth_is_schema_only(make_df, fails):
    """Standalone columns follow the same default as ``DataFrameSchema``."""
    column = pa.Column(int, pa.Check.gt(0), name="a", unique=True)
    assert column.validate(make_df({"a": [-1, -1]})) is not None
    fails(column, make_df({"a": ["x"]}), SchemaErrorReason.WRONG_DATATYPE)


def test_column_validation_disabled(make_df):
    invalid = make_df({"a": ["x"]})
    with config_context(validation_enabled=False):
        assert pa.Column(int, name="a").validate(invalid) is invalid


def test_column_coerce_is_not_supported(make_df, fails):
    """Column-level coerce is a documented gap in the narwhals backend."""
    schema = pa.DataFrameSchema({"a": pa.Column(str, coerce=True)})
    with pytest.warns(SchemaWarning, match="coerce=True is not applied"):
        fails(
            schema,
            make_df({"a": [1, 2, 3]}),
            SchemaErrorReason.WRONG_DATATYPE,
        )


def test_column_properties():
    column = pa.Column(
        int,
        name="a",
        nullable=True,
        unique=True,
        required=False,
        title="A",
        description="column a",
        metadata={"k": "v"},
    )
    properties = column.properties
    assert properties["name"] == "a"
    assert properties["nullable"]
    assert properties["unique"]
    assert not properties["required"]
    assert properties["title"] == "A"
    assert properties["description"] == "column a"
    assert properties["metadata"] == {"k": "v"}


def test_column_data_synthesis_not_supported():
    column = pa.Column(int, name="a")
    for method in (column.example, column.strategy, column.strategy_component):
        with pytest.raises(NotImplementedError, match="DataFusion"):
            method()
