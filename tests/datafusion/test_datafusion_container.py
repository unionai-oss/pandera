"""Tests for the DataFusion DataFrameSchema API."""

import datafusion
import pyarrow
import pytest
from datafusion import col, lit

import pandera.datafusion as pa
from pandera.api.datafusion.utils import get_validation_depth
from pandera.config import ValidationDepth, config_context
from pandera.errors import SchemaDefinitionError, SchemaErrorReason


@pytest.fixture
def data():
    return {
        "int_col": [1, 2, 3],
        "float_col": [1.0, 2.0, 3.0],
        "str_col": ["a", "b", "c"],
    }


@pytest.fixture
def df(make_df, data):
    return make_df(data)


@pytest.fixture
def schema():
    return pa.DataFrameSchema(
        {
            "int_col": pa.Column(int, pa.Check.gt(0)),
            "float_col": pa.Column(float),
            "str_col": pa.Column(str),
        }
    )


# Basic validation


def test_validate_returns_datafusion_dataframe(df, data, schema):
    validated = schema.validate(df)
    assert isinstance(validated, datafusion.DataFrame)
    assert validated.to_pydict() == data


def test_data_level_check_failure(make_df, data, schema, fails):
    error = fails(
        schema,
        make_df({**data, "int_col": [1, -2, 3]}),
        SchemaErrorReason.DATAFRAME_CHECK,
    )
    assert error.failure_cases.column("int_col").to_pylist() == [-2]


def test_wrong_dtype(make_df, data, schema, fails):
    fails(
        schema,
        make_df({**data, "int_col": [1.5, 2.5, 3.5]}),
        SchemaErrorReason.WRONG_DATATYPE,
    )


def test_missing_column(make_df, schema, fails):
    fails(
        schema,
        make_df({"int_col": [1, 2, 3]}),
        SchemaErrorReason.COLUMN_NOT_IN_DATAFRAME,
    )


def test_empty_dataframe(make_df):
    schema = pa.DataFrameSchema({"a": pa.Column(int, pa.Check.gt(0))})
    empty = make_df(pyarrow.table({"a": pyarrow.array([], pyarrow.int64())}))
    assert schema.validate(empty).to_pydict() == {"a": []}


# Column presence and order


def test_strict_rejects_extra_column(df, fails):
    schema = pa.DataFrameSchema({"int_col": pa.Column(int)}, strict=True)
    fails(schema, df, SchemaErrorReason.COLUMN_NOT_IN_SCHEMA)


def test_strict_filter_drops_extra_columns(df):
    schema = pa.DataFrameSchema({"int_col": pa.Column(int)}, strict="filter")
    assert list(schema.validate(df).schema().names) == ["int_col"]


def test_optional_column(make_df, fails):
    schema = pa.DataFrameSchema(
        {"a": pa.Column(int), "b": pa.Column(int, required=False)}
    )
    assert schema.validate(make_df({"a": [1]})).to_pydict() == {"a": [1]}
    fails(
        schema,
        make_df({"a": [1], "b": ["x"]}),
        SchemaErrorReason.WRONG_DATATYPE,
    )


def test_ordered(make_df, fails):
    schema = pa.DataFrameSchema(
        {"a": pa.Column(int), "b": pa.Column(int)}, ordered=True
    )
    assert schema.validate(make_df({"a": [1], "b": [2]})) is not None
    fails(
        schema,
        make_df({"b": [2], "a": [1]}),
        SchemaErrorReason.COLUMN_NOT_ORDERED,
    )


def test_unique_column_names_is_accepted(make_df):
    schema = pa.DataFrameSchema(
        {"a": pa.Column(int)}, unique_column_names=True
    )
    assert schema.validate(make_df({"a": [1]})) is not None


@pytest.mark.xfail(
    reason="the narwhals backend calls ``.copy()`` on the pyarrow table it "
    "builds for missing columns",
    raises=AttributeError,
    strict=True,
)
def test_add_missing_columns(make_df):
    schema = pa.DataFrameSchema(
        {"a": pa.Column(int), "b": pa.Column(int, default=7)},
        add_missing_columns=True,
    )
    validated = schema.validate(make_df({"a": [1]}))
    assert validated.to_pydict() == {"a": [1], "b": [7]}


# Regex columns


def test_regex_column(make_df, fails):
    schema = pa.DataFrameSchema({"^val_.+$": pa.Column(int, regex=True)})
    df = make_df({"val_a": [1], "val_b": [2], "other": ["x"]})
    assert list(schema.validate(df).schema().names) == [
        "val_a",
        "val_b",
        "other",
    ]
    fails(
        schema,
        make_df({"val_a": [1], "val_b": ["x"]}),
        SchemaErrorReason.WRONG_DATATYPE,
    )


def test_regex_column_without_match(make_df, fails):
    required = pa.DataFrameSchema({r"^val_\d$": pa.Column(int, regex=True)})
    optional = pa.DataFrameSchema(
        {r"^val_\d$": pa.Column(int, regex=True, required=False)}
    )
    df = make_df({"other": [1]})
    assert optional.validate(df).to_pydict() == {"other": [1]}
    fails(required, df, SchemaErrorReason.INVALID_COLUMN_NAME)


# Nullability and uniqueness


def test_nullable(make_df, fails):
    df = make_df({"a": [1, None, 3]})
    nullable = pa.DataFrameSchema({"a": pa.Column(int, nullable=True)})
    assert nullable.validate(df).to_pydict() == {"a": [1, None, 3]}
    fails(
        pa.DataFrameSchema({"a": pa.Column(int)}),
        df,
        SchemaErrorReason.SERIES_CONTAINS_NULLS,
    )


def test_unique_column(make_df, fails):
    schema = pa.DataFrameSchema({"a": pa.Column(int, unique=True)})
    assert schema.validate(make_df({"a": [1, 2, 3]})) is not None
    fails(
        schema,
        make_df({"a": [1, 1, 3]}),
        SchemaErrorReason.SERIES_CONTAINS_DUPLICATES,
    )


def test_jointly_unique_columns(make_df, fails):
    schema = pa.DataFrameSchema(
        {"a": pa.Column(int), "b": pa.Column(int)}, unique=["a", "b"]
    )
    assert schema.validate(make_df({"a": [1, 1], "b": [1, 2]})) is not None
    fails(
        schema,
        make_df({"a": [1, 1], "b": [2, 2]}),
        SchemaErrorReason.DUPLICATES,
    )


def test_report_duplicates_warns():
    with pytest.warns(UserWarning, match="report_duplicates"):
        pa.DataFrameSchema(
            {"a": pa.Column(int)}, report_duplicates="exclude_first"
        )


# Dataframe-level checks


def test_dataframe_level_check(make_df, fails):
    schema = pa.DataFrameSchema(
        {"a": pa.Column(int), "b": pa.Column(int)},
        checks=pa.Check(lambda data: col("a") < col("b")),
    )
    assert schema.validate(make_df({"a": [1], "b": [2]})) is not None
    error = fails(
        schema,
        make_df({"a": [3], "b": [2]}),
        SchemaErrorReason.DATAFRAME_CHECK,
    )
    assert error.failure_cases.to_pydict() == {"a": [3], "b": [2]}


# Lazy validation and failure cases


def test_lazy_collects_all_errors(make_df, fails):
    schema = pa.DataFrameSchema(
        {
            "a": pa.Column(int, pa.Check.gt(10)),
            "b": pa.Column(str),
            "c": pa.Column(int),
        }
    )
    error = fails(
        schema,
        make_df({"a": [1, 2], "b": [1, 2]}),
        SchemaErrorReason.DATAFRAME_CHECK,
        SchemaErrorReason.WRONG_DATATYPE,
        SchemaErrorReason.COLUMN_NOT_IN_DATAFRAME,
        lazy=True,
    )
    assert len(error.schema_errors) == 3


def test_failure_cases_are_a_pyarrow_table(make_df, fails):
    """Failure cases are reported as pyarrow, DataFusion's native format."""
    schema = pa.DataFrameSchema({"a": pa.Column(int, pa.Check.gt(0))})
    failure_cases = fails(
        schema,
        make_df({"a": [1, -2, -3]}),
        SchemaErrorReason.DATAFRAME_CHECK,
        lazy=True,
    ).failure_cases

    assert isinstance(failure_cases, pyarrow.Table)
    assert set(failure_cases.column_names) >= {
        "failure_case",
        "schema_context",
        "column",
        "check",
        "check_number",
        "index",
    }
    assert failure_cases.column("failure_case").to_pylist() == ["-2", "-3"]
    assert failure_cases.column("column").to_pylist() == ["a", "a"]
    assert failure_cases.column("index").to_pylist() == [None, None]


def test_failure_cases_include_schema_level_errors(make_df, fails):
    """Scalar (schema-level) and row-level failure cases must concatenate."""
    schema = pa.DataFrameSchema(
        {
            "a": pa.Column(int, pa.Check.gt(0)),
            "missing": pa.Column(str),
        }
    )
    failure_cases = fails(
        schema,
        make_df({"a": [1, -2]}),
        SchemaErrorReason.DATAFRAME_CHECK,
        SchemaErrorReason.COLUMN_NOT_IN_DATAFRAME,
        lazy=True,
    ).failure_cases

    assert isinstance(failure_cases, pyarrow.Table)
    assert set(failure_cases.column("failure_case").to_pylist()) == {
        "missing",
        "-2",
    }
    assert set(failure_cases.column("schema_context").to_pylist()) == {
        "DataFrameSchema",
        "Column",
    }


@pytest.mark.parametrize(
    "column,data,reason",
    [
        (
            pa.Column(int),
            [1, None],
            SchemaErrorReason.SERIES_CONTAINS_NULLS,
        ),
        (
            pa.Column(int, unique=True),
            [1, 1],
            SchemaErrorReason.SERIES_CONTAINS_DUPLICATES,
        ),
    ],
)
def test_lazy_reports_nulls_and_duplicates(
    make_df, fails, column, data, reason
):
    schema = pa.DataFrameSchema({"a": column})
    error = fails(schema, make_df({"a": data}), reason, lazy=True)
    assert isinstance(error.failure_cases, pyarrow.Table)


@pytest.mark.xfail(
    reason="scalar failure cases are built with polars whenever it is "
    "installed, so the type depends on the environment",
    strict=True,
)
def test_schema_level_failure_cases_are_a_pyarrow_table(make_df, fails):
    pytest.importorskip("polars")
    schema = pa.DataFrameSchema({"a": pa.Column(str), "b": pa.Column(int)})
    error = fails(
        schema,
        make_df({"a": [1]}),
        SchemaErrorReason.WRONG_DATATYPE,
        SchemaErrorReason.COLUMN_NOT_IN_DATAFRAME,
        lazy=True,
    )
    assert isinstance(error.failure_cases, pyarrow.Table)


# Dropping invalid rows


def test_drop_invalid_rows(make_df):
    schema = pa.DataFrameSchema(
        {"a": pa.Column(int, pa.Check.gt(0))}, drop_invalid_rows=True
    )
    validated = schema.validate(make_df({"a": [1, -2, 3]}), lazy=True)
    assert isinstance(validated, datafusion.DataFrame)
    assert validated.to_pydict() == {"a": [1, 3]}


def test_drop_invalid_rows_drops_nulls(make_df):
    schema = pa.DataFrameSchema({"a": pa.Column(int)}, drop_invalid_rows=True)
    validated = schema.validate(make_df({"a": [1, None]}), lazy=True)
    assert validated.to_pydict() == {"a": [1]}


def test_drop_invalid_rows_requires_lazy(make_df):
    schema = pa.DataFrameSchema(
        {"a": pa.Column(int, pa.Check.gt(0))}, drop_invalid_rows=True
    )
    with pytest.raises(SchemaDefinitionError, match="lazy must be set"):
        schema.validate(make_df({"a": [1, -2]}))


@pytest.mark.usefixtures("default_depth")
def test_drop_invalid_rows_runs_data_checks_at_default_depth(make_df):
    schema = pa.DataFrameSchema(
        {"a": pa.Column(int, pa.Check.gt(0))}, drop_invalid_rows=True
    )
    validated = schema.validate(make_df({"a": [1, -2]}), lazy=True)
    assert validated.to_pydict() == {"a": [1]}


def test_drop_invalid_rows_ignores_native_checks(make_df):
    """Documented limitation: expression checks do not drop rows."""
    schema = pa.DataFrameSchema(
        {"a": pa.Column(int, pa.Check(lambda d: col(d.key) > lit(0)))},
        drop_invalid_rows=True,
    )
    validated = schema.validate(make_df({"a": [1, -2, 3]}), lazy=True)
    assert validated.to_pydict() == {"a": [1, -2, 3]}


def test_drop_invalid_rows_keeps_duplicates(make_df):
    """Documented narwhals limitation: uniqueness failures are not dropped."""
    schema = pa.DataFrameSchema(
        {"a": pa.Column(int, unique=True)}, drop_invalid_rows=True
    )
    validated = schema.validate(make_df({"a": [1, 1, 2]}), lazy=True)
    assert validated.to_pydict() == {"a": [1, 1, 2]}


# Validation depth and laziness


@pytest.mark.usefixtures("default_depth")
def test_default_validation_depth_is_schema_only(df):
    """A datafusion.DataFrame is lazy: data checks are opt-in, like ibis."""
    assert get_validation_depth(df) is ValidationDepth.SCHEMA_ONLY
    with config_context(validation_depth=ValidationDepth.SCHEMA_AND_DATA):
        assert get_validation_depth(df) is ValidationDepth.SCHEMA_AND_DATA


@pytest.mark.usefixtures("default_depth")
def test_default_depth_skips_data_checks(make_df, data, schema, fails):
    invalid = make_df({**data, "int_col": [1, -2, 3]})
    assert isinstance(schema.validate(invalid), datafusion.DataFrame)
    fails(
        schema,
        make_df({**data, "int_col": ["x", "y", "z"]}),
        SchemaErrorReason.WRONG_DATATYPE,
    )


def test_schema_only_depth_skips_data_checks(make_df, data, schema):
    invalid = make_df({**data, "int_col": [1, -2, 3]})
    with config_context(validation_depth=ValidationDepth.SCHEMA_ONLY):
        assert isinstance(schema.validate(invalid), datafusion.DataFrame)


@pytest.mark.usefixtures("default_depth")
def test_schema_only_validation_does_not_execute_the_plan(
    df, make_df, data, schema, fails, executions
):
    schema.validate(df)
    fails(
        schema,
        make_df({**data, "int_col": ["x", "y", "z"]}),
        SchemaErrorReason.WRONG_DATATYPE,
    )
    assert executions == []


@pytest.mark.xfail(
    reason="the missing-column error message previews the first rows, which "
    "executes the plan",
    strict=True,
)
@pytest.mark.usefixtures("default_depth")
def test_missing_column_error_does_not_execute_the_plan(
    make_df, schema, fails, executions
):
    fails(
        schema,
        make_df({"int_col": [1, 2, 3]}),
        SchemaErrorReason.COLUMN_NOT_IN_DATAFRAME,
    )
    assert executions == []


def test_data_checks_return_only_aggregates(make_df, executions):
    """Passing data-level validation never pulls rows into Python."""
    schema = pa.DataFrameSchema(
        {"a": pa.Column(int, pa.Check.gt(0), unique=True)}
    )
    schema.validate(make_df({"a": list(range(1, 101))}))
    assert executions
    assert max(executions) == 1


def test_failing_check_collects_only_failing_rows(make_df, fails, executions):
    schema = pa.DataFrameSchema({"a": pa.Column(int, pa.Check.gt(0))})
    fails(
        schema,
        make_df({"a": [-5, *range(1, 100)]}),
        SchemaErrorReason.DATAFRAME_CHECK,
    )
    assert max(executions) == 1


def test_lazy_null_report_collects_the_whole_dataframe(
    make_df, fails, executions
):
    """Documented limitation of ``lazy=True`` reporting."""
    schema = pa.DataFrameSchema({"a": pa.Column(int)})
    fails(
        schema,
        make_df({"a": [None, *range(1, 100)]}),
        SchemaErrorReason.SERIES_CONTAINS_NULLS,
        lazy=True,
    )
    assert max(executions) == 100


def test_validation_disabled(make_df, data, schema):
    invalid = make_df({**data, "int_col": [-1, -2, -3]})
    with config_context(validation_enabled=False):
        assert schema.validate(invalid) is invalid


# Row sampling


def test_head_subsample(make_df, fails):
    schema = pa.DataFrameSchema({"a": pa.Column(int, pa.Check.gt(0))})
    df = make_df({"a": [1, 2, -3]})
    assert schema.validate(df, head=2).to_pydict() == {"a": [1, 2, -3]}
    fails(schema, df, SchemaErrorReason.DATAFRAME_CHECK)


@pytest.mark.parametrize("kwargs", [{"tail": 1}, {"sample": 1}])
def test_tail_and_sample_are_not_supported(make_df, kwargs):
    schema = pa.DataFrameSchema({"a": pa.Column(int)})
    with pytest.raises(NotImplementedError, match="is not supported"):
        schema.validate(make_df({"a": [1, 2]}), **kwargs)


# Unsupported features


def test_schema_level_coerce_is_silently_ignored(make_df, fails, recwarn):
    """Documented limitation: no cast and, unlike column level, no warning."""
    schema = pa.DataFrameSchema({"a": pa.Column(str)}, coerce=True)
    fails(schema, make_df({"a": [1, 2]}), SchemaErrorReason.WRONG_DATATYPE)
    assert not recwarn.list


def test_data_synthesis_not_supported(schema):
    for method in (schema.example, schema.strategy):
        with pytest.raises(NotImplementedError, match="DataFusion"):
            method()


# Schema transformations


def test_schema_transformations_keep_the_datafusion_schema(make_df, fails):
    schema = pa.DataFrameSchema({"a": pa.Column(int), "b": pa.Column(str)})

    added = schema.add_columns({"c": pa.Column(float)})
    removed = schema.remove_columns(["b"])
    renamed = schema.rename_columns({"a": "z"})
    updated = schema.update_column("a", nullable=True)
    selected = schema.select_columns(["b"])

    for transformed in (added, removed, renamed, updated, selected):
        assert isinstance(transformed, pa.DataFrameSchema)
    assert list(added.columns) == ["a", "b", "c"]
    assert list(removed.columns) == ["a"]
    assert list(renamed.columns) == ["z", "b"]
    assert updated.columns["a"].nullable
    assert list(selected.columns) == ["b"]

    df = make_df({"a": [1, None], "b": ["x", "y"]})
    assert updated.validate(df) is not None
    fails(added, df, SchemaErrorReason.COLUMN_NOT_IN_DATAFRAME)


def test_schema_equality():
    def build():
        return pa.DataFrameSchema(
            {"a": pa.Column(int, pa.Check.gt(0)), "b": pa.Column(str)}
        )

    assert build() == build()
    assert build() != build().remove_columns(["b"])
