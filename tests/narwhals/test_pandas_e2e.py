"""End-to-end tests for the narwhals backend on pandas DataFrames.

Cross-backend behavior shared with polars/ibis is covered by the parametrized
suite in tests/narwhals/backends/. This module covers pandas-specific
behavior: native dtype semantics, pandas-style check functions, NaN-based
ignore_na handling, failure-case frame types, and the index-validation gap.

Requires PANDERA_USE_NARWHALS_BACKEND=True (set by the narwhals nox session);
the fixture below enforces it for local runs.
"""

import warnings

import pandas as pd
import pytest

pytest.importorskip("narwhals")

import pandera.pandas as pa
from pandera.backends.narwhals.container import (
    DataFrameSchemaBackend as NarwhalsDataFrameSchemaBackend,
)
from pandera.errors import SchemaError, SchemaErrors, SchemaWarning


@pytest.fixture(autouse=True)
def use_narwhals_backend():
    """Force narwhals-backed pandas registration for every test here."""
    from pandera.backends.pandas.register import register_pandas_backends
    from pandera.config import CONFIG

    original_flag = CONFIG.use_narwhals_backend
    CONFIG.use_narwhals_backend = True
    register_pandas_backends.cache_clear()
    yield
    CONFIG.use_narwhals_backend = original_flag
    register_pandas_backends.cache_clear()
    from pandera.backends.narwhals.register import (
        clear_narwhals_compatible_backend_registry,
    )

    clear_narwhals_compatible_backend_registry()


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame(
        {"x": [1, 2, 3], "s": ["a", "b", "c"], "f": [0.5, 1.5, 2.5]}
    )


def test_narwhals_backend_is_used(df):
    schema = pa.DataFrameSchema({"x": pa.Column(int)})
    assert isinstance(schema.get_backend(df), NarwhalsDataFrameSchemaBackend)


def test_validate_returns_pandas_dataframe(df):
    schema = pa.DataFrameSchema({"x": pa.Column(int)})
    out = schema.validate(df)
    assert isinstance(out, pd.DataFrame)
    pd.testing.assert_frame_equal(out, df)


def test_validate_preserves_index():
    df = pd.DataFrame({"x": [1, 2]}, index=["a", "b"])
    out = pa.DataFrameSchema({"x": pa.Column(int)}).validate(df)
    assert list(out.index) == ["a", "b"]


# ---------------------------------------------------------------------------
# dtype semantics — native pandas engine comparisons
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype,data",
    [
        (int, [1, 2]),
        ("int64", [1, 2]),
        (float, [1.0, 2.0]),
        (str, ["a", "b"]),  # object dtype must satisfy str schema
        (object, ["a", 1]),
        (bool, [True, False]),
        ("datetime64[ns]", pd.to_datetime(["2021-01-01", "2021-01-02"])),
        ("category", pd.Categorical(["a", "b"])),
        ("Int64", pd.array([1, 2], dtype="Int64")),
    ],
)
def test_dtype_passes(dtype, data):
    frame = pd.DataFrame({"col": data})
    out = pa.DataFrameSchema({"col": pa.Column(dtype)}).validate(frame)
    assert isinstance(out, pd.DataFrame)


@pytest.mark.parametrize(
    "dtype,data",
    [
        (int, ["a", "b"]),
        (str, [1, 2]),
        ("int32", [1, 2]),  # int64 data — width must be respected
        (float, [1, 2]),
        ("datetime64[ns]", [1, 2]),
    ],
)
def test_dtype_mismatch_raises(dtype, data):
    frame = pd.DataFrame({"col": data})
    with pytest.raises(SchemaError, match="expected column 'col'"):
        pa.DataFrameSchema({"col": pa.Column(dtype)}).validate(frame)


def test_dtype_failure_cases_report_native_dtype():
    with pytest.raises(SchemaError) as exc_info:
        pa.DataFrameSchema({"x": pa.Column(str)}).validate(
            pd.DataFrame({"x": [1, 2]})
        )
    assert exc_info.value.failure_cases == "int64"


# ---------------------------------------------------------------------------
# pandas-style check functions keep working under the narwhals backend
# ---------------------------------------------------------------------------


def test_series_check_passes(df):
    schema = pa.DataFrameSchema(
        {"x": pa.Column(int, pa.Check(lambda s: s > 0))}
    )
    assert isinstance(schema.validate(df), pd.DataFrame)


def test_series_check_fails_with_pandas_failure_cases(df):
    schema = pa.DataFrameSchema(
        {"x": pa.Column(int, pa.Check(lambda s: s > 1))}
    )
    with pytest.raises(SchemaError) as exc_info:
        schema.validate(df)
    failure_cases = exc_info.value.failure_cases
    assert isinstance(failure_cases, pd.DataFrame)
    assert failure_cases["x"].tolist() == [1]


def test_aggregate_check_returning_numpy_bool(df):
    schema = pa.DataFrameSchema(
        {"x": pa.Column(int, pa.Check(lambda s: (s > 0).all()))}
    )
    assert isinstance(schema.validate(df), pd.DataFrame)

    failing = pa.DataFrameSchema(
        {"x": pa.Column(int, pa.Check(lambda s: (s > 1).all()))}
    )
    with pytest.raises(SchemaError):
        failing.validate(df)


def test_check_returning_numpy_array(df):
    schema = pa.DataFrameSchema(
        {"x": pa.Column(int, pa.Check(lambda s: s.to_numpy() > 0))}
    )
    assert isinstance(schema.validate(df), pd.DataFrame)


def test_dataframe_level_check(df):
    schema = pa.DataFrameSchema(
        {"x": pa.Column(int)},
        checks=pa.Check(lambda d: d["x"] > 0),
    )
    assert isinstance(schema.validate(df), pd.DataFrame)

    failing = pa.DataFrameSchema(
        {"x": pa.Column(int)},
        checks=pa.Check(lambda d: d["x"] > 1),
    )
    with pytest.raises(SchemaError):
        failing.validate(df)


def test_element_wise_check(df):
    schema = pa.DataFrameSchema(
        {"x": pa.Column(int, pa.Check(lambda v: v > 0, element_wise=True))}
    )
    assert isinstance(schema.validate(df), pd.DataFrame)


def test_narwhals_expression_check(df):
    """native=False checks receive a narwhals column expression."""
    schema = pa.DataFrameSchema(
        {"x": pa.Column(int, pa.Check(lambda col: col < 100, native=False))}
    )
    assert isinstance(schema.validate(df), pd.DataFrame)


def test_builtin_checks(df):
    schema = pa.DataFrameSchema(
        {
            "x": pa.Column(int, [pa.Check.ge(0), pa.Check.isin([1, 2, 3])]),
            "s": pa.Column(str, pa.Check.str_matches(r"^[abc]$")),
            "f": pa.Column(float, pa.Check.in_range(0, 3)),
        }
    )
    assert isinstance(schema.validate(df), pd.DataFrame)


# ---------------------------------------------------------------------------
# ignore_na semantics with NaN-based missing values
# ---------------------------------------------------------------------------


def test_ignore_na_passes_nan_rows():
    """NaN rows pass checks by default (ignore_na=True), like the native backend."""
    frame = pd.DataFrame({"a": [None, 1.0, 2.0]})
    schema = pa.DataFrameSchema(
        {"a": pa.Column(float, pa.Check.ge(0), nullable=True)}
    )
    assert isinstance(schema.validate(frame), pd.DataFrame)


def test_ignore_na_false_fails_nan_rows():
    frame = pd.DataFrame({"a": [None, 1.0, 2.0]})
    schema = pa.DataFrameSchema(
        {"a": pa.Column(float, pa.Check.ge(0, ignore_na=False), nullable=True)}
    )
    with pytest.raises(SchemaError):
        schema.validate(frame)


def test_ignore_na_with_nullable_extension_dtype():
    frame = pd.DataFrame({"a": pd.array([None, 1, 2], dtype="Int64")})
    schema = pa.DataFrameSchema(
        {"a": pa.Column("Int64", pa.Check.ge(0), nullable=True)}
    )
    assert isinstance(schema.validate(frame), pd.DataFrame)


def test_drop_invalid_rows_keeps_nan_rows():
    """drop_invalid_rows must not drop NaN rows when the check ignores NA."""
    frame = pd.DataFrame({"a": [None, -1.0, 0.0, 1.0]})
    schema = pa.DataFrameSchema(
        {"a": pa.Column(float, pa.Check.ge(0), nullable=True)},
        drop_invalid_rows=True,
    )
    result = schema.validate(frame, lazy=True)
    assert result["a"].isna().sum() == 1
    assert result["a"].dropna().tolist() == [0.0, 1.0]


# ---------------------------------------------------------------------------
# failure cases and lazy validation
# ---------------------------------------------------------------------------


def test_lazy_failure_cases_are_pandas(df):
    """Aggregated failure_cases frame is pandas, even for scalar-only errors."""
    schema = pa.DataFrameSchema(
        {
            "x": pa.Column(str),  # scalar failure case (wrong dtype)
            "s": pa.Column(str, pa.Check.isin(["a"])),  # data failure cases
            "missing": pa.Column(int),  # scalar failure case (absent column)
        }
    )
    with pytest.raises(SchemaErrors) as exc_info:
        schema.validate(df, lazy=True)

    err = exc_info.value
    assert len(err.schema_errors) == 3
    assert isinstance(err.failure_cases, pd.DataFrame)
    assert set(err.failure_cases.columns) == {
        "failure_case",
        "schema_context",
        "column",
        "check",
        "check_number",
        "index",
    }
    # data-level failure cases contain the failing values, not JSON structs
    isin_cases = err.failure_cases.query("column == 's'")
    assert sorted(isin_cases["failure_case"]) == ["b", "c"]


def test_lazy_failure_cases_scalar_only(df):
    """All-scalar failure cases (dtype errors) also produce a pandas frame."""
    schema = pa.DataFrameSchema({"x": pa.Column(str), "s": pa.Column(int)})
    with pytest.raises(SchemaErrors) as exc_info:
        schema.validate(df, lazy=True)
    failure_cases = exc_info.value.failure_cases
    assert isinstance(failure_cases, pd.DataFrame)
    assert sorted(failure_cases["failure_case"]) == ["int64", "object"]


def test_schema_error_failure_cases_no_index_leak(df):
    """Failure-case frames must not contain pandas index artifacts."""
    schema = pa.DataFrameSchema({"s": pa.Column(str, pa.Check.isin(["a"]))})
    with pytest.raises(SchemaErrors) as exc_info:
        schema.validate(df, lazy=True)
    failure_cases = exc_info.value.failure_cases
    assert not any(
        "__index_level_" in str(v) for v in failure_cases["failure_case"]
    )


# ---------------------------------------------------------------------------
# pandas-specific schema features
# ---------------------------------------------------------------------------


def test_index_component_warns_and_is_skipped(df):
    schema = pa.DataFrameSchema({"x": pa.Column(int)}, index=pa.Index(str))
    with pytest.warns(
        SchemaWarning, match="index validation is not supported"
    ):
        out = schema.validate(df.set_index("s"))
    assert isinstance(out, pd.DataFrame)


def test_column_coerce_warns(df):
    schema = pa.DataFrameSchema({"x": pa.Column(float, coerce=True)})
    with pytest.warns(SchemaWarning, match="coerce=True is not applied"):
        with pytest.raises(SchemaError):
            schema.validate(df)


def test_series_schema_stays_native():
    """SeriesSchema validation is unaffected by the narwhals backend flag."""
    schema = pa.SeriesSchema(int, pa.Check.ge(0))
    out = schema.validate(pd.Series([1, 2, 3]))
    assert isinstance(out, pd.Series)


def test_dataframe_model(df):
    class Model(pa.DataFrameModel):
        x: int
        s: str
        f: float

    out = Model.validate(df)
    assert isinstance(out, pd.DataFrame)

    class FailingModel(pa.DataFrameModel):
        x: str

    with pytest.raises(SchemaError):
        FailingModel.validate(df)


def test_regex_columns():
    frame = pd.DataFrame({"col_a": [1], "col_b": [2], "other": ["x"]})
    schema = pa.DataFrameSchema(
        {"^col_.*$": pa.Column(int, pa.Check.ge(0), regex=True)}
    )
    out = schema.validate(frame)
    assert isinstance(out, pd.DataFrame)

    with pytest.raises(SchemaError):
        schema.validate(pd.DataFrame({"col_a": [-1], "other": ["x"]}))


def test_ordered_schema():
    frame = pd.DataFrame({"b": [1], "a": [2]})
    schema = pa.DataFrameSchema(
        {"a": pa.Column(int), "b": pa.Column(int)}, ordered=True
    )
    with pytest.raises(SchemaError, match="out-of-order"):
        schema.validate(frame)


def test_pandas_column_selector_property():
    """The pandas Column exposes the selector property used by the backend."""
    assert pa.Column(int, name="a").selector == "a"
    assert pa.Column(int, name="a", regex=True).selector == "^a$"
    assert pa.Column(int, name="^a.*$", regex=True).selector == "^a.*$"
    assert pa.Column(int, name=("a", "b")).selector == ("a", "b")
