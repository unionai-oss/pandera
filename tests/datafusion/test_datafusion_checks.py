"""Tests for checks against DataFusion DataFrames."""

import pyarrow
import pytest
from datafusion import col, lit
from datafusion import functions as f

import pandera.datafusion as pa
from pandera.api.datafusion.types import DataFusionData
from pandera.constants import CHECK_OUTPUT_KEY
from pandera.errors import SchemaErrorReason, SchemaWarning


def _schema(check, dtype=int):
    return pa.DataFrameSchema({"a": pa.Column(dtype, check)})


# Built-in checks


@pytest.mark.parametrize(
    "check,passing,failing",
    [
        (pa.Check.gt(0), [1, 2], [0, 1]),
        (pa.Check.ge(0), [0, 1], [-1, 0]),
        (pa.Check.lt(10), [1, 2], [10, 1]),
        (pa.Check.le(10), [10], [11]),
        (pa.Check.eq(1), [1, 1], [1, 2]),
        (pa.Check.ne(1), [2, 3], [1, 2]),
        (pa.Check.isin([1, 2]), [1, 2], [1, 3]),
        (pa.Check.notin([3]), [1, 2], [3]),
        (pa.Check.between(0, 5), [1, 5], [6]),
    ],
)
def test_builtin_numeric_checks(make_df, fails, check, passing, failing):
    schema = _schema(check)
    assert schema.validate(make_df({"a": passing})) is not None
    fails(schema, make_df({"a": failing}), SchemaErrorReason.DATAFRAME_CHECK)


@pytest.mark.parametrize(
    "check,passing,failing",
    [
        (pa.Check.str_startswith("a"), ["ab", "ac"], ["ab", "bc"]),
        (pa.Check.str_endswith("z"), ["az"], ["za"]),
        (pa.Check.str_contains("b"), ["abc"], ["acd"]),
        (pa.Check.str_matches(r"^\d+$"), ["123"], ["12a"]),
        (pa.Check.str_length(2, 3), ["ab", "abc"], ["a"]),
    ],
)
def test_builtin_string_checks(make_df, fails, check, passing, failing):
    schema = _schema(check, dtype=str)
    assert schema.validate(make_df({"a": passing})) is not None
    fails(schema, make_df({"a": failing}), SchemaErrorReason.DATAFRAME_CHECK)


def test_multiple_checks_on_a_column(make_df, fails):
    schema = pa.DataFrameSchema(
        {"a": pa.Column(int, [pa.Check.gt(0), pa.Check.lt(10)])}
    )
    assert schema.validate(make_df({"a": [1, 9]})) is not None
    error = fails(
        schema,
        make_df({"a": [0, 10]}),
        SchemaErrorReason.DATAFRAME_CHECK,
        lazy=True,
    )
    assert len(error.schema_errors) == 2


# Check options


def test_failure_cases_are_reported(make_df, fails):
    error = fails(
        _schema(pa.Check.gt(0)),
        make_df({"a": [1, -2, -3]}),
        SchemaErrorReason.DATAFRAME_CHECK,
    )
    assert error.failure_cases.column("a").to_pylist() == [-2, -3]


def test_custom_error_message(make_df, fails):
    error = fails(
        _schema(pa.Check.gt(0, error="must be positive")),
        make_df({"a": [-1]}),
        SchemaErrorReason.DATAFRAME_CHECK,
    )
    assert "must be positive" in str(error)


def test_raise_warning_does_not_fail_validation(make_df):
    schema = _schema(pa.Check.gt(0, raise_warning=True))
    with pytest.warns(SchemaWarning, match="greater_than"):
        validated = schema.validate(make_df({"a": [-1]}))
    assert validated.to_pydict() == {"a": [-1]}


def test_n_failure_cases_limits_the_report(make_df, fails):
    error = fails(
        _schema(pa.Check.gt(0, n_failure_cases=1)),
        make_df({"a": [-1, -2, -3]}),
        SchemaErrorReason.DATAFRAME_CHECK,
    )
    assert error.failure_cases.num_rows == 1


def test_ignore_na_skips_nulls(make_df):
    schema = pa.DataFrameSchema(
        {"a": pa.Column(int, pa.Check.gt(0), nullable=True)}
    )
    assert schema.validate(make_df({"a": [1, None]})) is not None


@pytest.mark.xfail(
    reason="the narwhals backend ignores null check outputs even when "
    "ignore_na=False",
    strict=True,
)
def test_ignore_na_false_fails_on_nulls(make_df, fails):
    schema = pa.DataFrameSchema(
        {"a": pa.Column(int, pa.Check.gt(0, ignore_na=False), nullable=True)}
    )
    fails(
        schema,
        make_df({"a": [1, None]}),
        SchemaErrorReason.DATAFRAME_CHECK,
    )


# Custom checks: calling conventions


def test_native_check_datafusion_data_convention(make_df, fails):
    """A 1-arg native check receives a ``DataFusionData`` container."""
    seen = {}

    def check_fn(data):
        seen["type"] = type(data)
        seen["key"] = data.key
        seen["dataframe"] = data.dataframe
        return col(data.key) > lit(0)

    schema = _schema(pa.Check(check_fn))
    df = make_df({"a": [1, 2]})
    schema.validate(df)

    assert seen["type"] is DataFusionData
    assert seen["key"] == "a"
    assert seen["dataframe"].to_pydict() == df.to_pydict()

    fails(schema, make_df({"a": [1, -2]}), SchemaErrorReason.DATAFRAME_CHECK)


def test_dataframe_level_check_key_is_star(make_df):
    seen = {}

    def check_fn(data):
        seen["key"] = data.key
        return col("a") > lit(0)

    schema = pa.DataFrameSchema(
        {"a": pa.Column(int)}, checks=pa.Check(check_fn)
    )
    schema.validate(make_df({"a": [1]}))
    assert seen["key"] == "*"


def test_native_check_two_arg_convention(make_df, fails):
    """A 2-arg native check gets ``(native_dataframe, key)``."""
    schema = _schema(pa.Check(lambda df, key: col(key) > lit(0)))
    assert schema.validate(make_df({"a": [1, 2]})) is not None
    fails(schema, make_df({"a": [1, -2]}), SchemaErrorReason.DATAFRAME_CHECK)


def test_non_native_expression_check(make_df, fails):
    """``native=False`` uses the narwhals expression protocol."""
    schema = _schema(pa.Check(lambda c: c > 0, native=False))
    assert schema.validate(make_df({"a": [1, 2]})) is not None
    error = fails(
        schema, make_df({"a": [1, -2]}), SchemaErrorReason.DATAFRAME_CHECK
    )
    assert error.failure_cases.column("a").to_pylist() == [-2]


# Custom checks: return types


def test_native_check_reports_failing_rows(make_df, fails):
    """A row-level ``Expr`` keeps the offending values as failure cases."""
    error = fails(
        _schema(pa.Check(lambda d: col(d.key) > lit(0))),
        make_df({"a": [1, -2]}),
        SchemaErrorReason.DATAFRAME_CHECK,
    )
    assert error.failure_cases.column("a").to_pylist() == [-2]


@pytest.mark.parametrize("column_level", [True, False])
def test_native_check_returning_aggregate_expr(make_df, fails, column_level):
    """An aggregate ``Expr`` is evaluated to a single pass/fail result."""
    check = pa.Check(lambda d: f.max(col("a")) < lit(10))
    schema = (
        _schema(check)
        if column_level
        else pa.DataFrameSchema({"a": pa.Column(int)}, checks=check)
    )
    assert schema.validate(make_df({"a": [1, 5, 9]})) is not None
    fails(
        schema, make_df({"a": [1, 5, 99]}), SchemaErrorReason.DATAFRAME_CHECK
    )


def test_aggregate_expr_respects_head(make_df):
    schema = _schema(pa.Check(lambda d: f.max(col(d.key)) < lit(10)))
    assert schema.validate(make_df({"a": [1, 5, 99]}), head=2) is not None


def test_aggregate_expr_on_empty_dataframe_fails(make_df, fails):
    """A null aggregate result counts as not passed."""
    empty = make_df(pyarrow.table({"a": pyarrow.array([], pyarrow.int64())}))
    fails(
        _schema(pa.Check(lambda d: f.max(col(d.key)) < lit(10))),
        empty,
        SchemaErrorReason.DATAFRAME_CHECK,
    )


def test_native_check_returning_boolean(make_df, fails):
    """A python bool is accepted as an aggregate result."""
    schema = _schema(
        pa.Check(lambda df, key: df.filter(col(key) <= lit(0)).count() == 0)
    )
    assert schema.validate(make_df({"a": [1, 2]})) is not None
    fails(schema, make_df({"a": [1, -2]}), SchemaErrorReason.DATAFRAME_CHECK)


@pytest.mark.parametrize(
    "check_fn",
    [
        lambda d: d.dataframe.select(col(d.key) > lit(0)),
        lambda d: d.dataframe.with_column(CHECK_OUTPUT_KEY, lit(True)),
    ],
)
def test_native_check_returning_dataframe_is_rejected(
    make_df, fails, check_fn
):
    """A lazy DataFrame cannot be aligned row-wise with the validated one."""
    error = fails(
        _schema(pa.Check(check_fn)),
        make_df({"a": [1]}),
        SchemaErrorReason.CHECK_ERROR,
    )
    assert "datafusion.Expr" in str(error)


def test_native_check_invalid_expr_reports_original_error(make_df, fails):
    """A row-level planning error is not masked by the aggregate fallback."""
    error = fails(
        _schema(pa.Check(lambda d: f.abs(col(d.key)) > lit(0)), dtype=str),
        make_df({"a": ["x"]}),
        SchemaErrorReason.CHECK_ERROR,
    )
    assert "expects Numeric" in str(error)


def test_native_check_mixing_aggregate_and_row_values_fails(make_df, fails):
    """Neither a row-level nor an aggregate expression: reported as an error."""
    fails(
        _schema(pa.Check(lambda d: f.max(col(d.key)) < col(d.key))),
        make_df({"a": [1, 2]}),
        SchemaErrorReason.CHECK_ERROR,
    )


def test_element_wise_check_is_not_supported(make_df, fails):
    error = fails(
        _schema(pa.Check(lambda v: v > 0, element_wise=True)),
        make_df({"a": [1, 2]}),
        SchemaErrorReason.CHECK_ERROR,
    )
    assert "element_wise checks are not supported" in str(error)
