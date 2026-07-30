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


def test_dataframe_check_returning_bool_dataframe(df):
    """Multi-column boolean DataFrame outputs are AND-reduced per row."""
    schema = pa.DataFrameSchema(
        {"x": pa.Column(int)},
        checks=pa.Check(lambda d: d[["x", "f"]] > 0),
    )
    assert isinstance(schema.validate(df), pd.DataFrame)

    failing = pa.DataFrameSchema(
        {"x": pa.Column(int)},
        checks=pa.Check(lambda d: d[["x", "f"]] > 1),
    )
    with pytest.raises(SchemaError):
        failing.validate(df)


def test_check_returning_dataframe_with_check_output_key(df):
    """DataFrame outputs carrying CHECK_OUTPUT_KEY use that column directly."""
    from pandera.constants import CHECK_OUTPUT_KEY

    schema = pa.DataFrameSchema(
        {
            "x": pa.Column(
                int,
                pa.Check(lambda s: pd.DataFrame({CHECK_OUTPUT_KEY: s > 0})),
            )
        }
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


def test_drop_invalid_rows_nullable_violation():
    """drop_invalid_rows removes rows violating nullable=False."""
    frame = pd.DataFrame({"a": [1.0, None, 2.0]})
    schema = pa.DataFrameSchema(
        {"a": pa.Column(float, nullable=False)},
        drop_invalid_rows=True,
    )
    result = schema.validate(frame, lazy=True)
    assert result["a"].tolist() == [1.0, 2.0]


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
    # string columns are dtype "object" on pandas < 3 and "str" on pandas >= 3
    assert sorted(failure_cases["failure_case"]) == sorted(
        ["int64", str(df["s"].dtype)]
    )


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


def test_index_validation(df):
    """The narwhals backend validates pandas Index components (delegated to
    the native Index backend) and preserves the index in the output."""
    schema = pa.DataFrameSchema(
        {"x": pa.Column(int)}, index=pa.Index(str, name="s")
    )
    out = schema.validate(df.set_index("s"))
    assert isinstance(out, pd.DataFrame)
    assert out.index.name == "s"
    assert list(out.index) == list(df["s"])


def test_index_validation_failure(df):
    schema = pa.DataFrameSchema(
        {"x": pa.Column(int)},
        index=pa.Index(int, pa.Check.ge(0), name="idx"),
    )
    bad = df.copy()
    bad.index = pd.Index([-1, 0, 1], name="idx")
    with pytest.raises(SchemaError):
        schema.validate(bad)
    with pytest.raises(pa.errors.SchemaErrors) as exc_info:
        schema.validate(bad, lazy=True)
    # the failure is reported against the index check with the offending value
    fcs = exc_info.value.failure_cases["failure_case"].astype(str).str.cat()
    assert "-1" in fcs
    assert any(
        "greater_than_or_equal_to" in str(c)
        for c in exc_info.value.failure_cases["check"]
    )


def test_multiindex_validation():
    schema = pa.DataFrameSchema(
        {"a": pa.Column(int)},
        index=pa.MultiIndex(
            [
                pa.Index(int, name="i0"),
                pa.Index(str, pa.Check.isin(["x", "y"]), name="i1"),
            ]
        ),
    )
    mi = pd.MultiIndex.from_tuples([(0, "x"), (1, "y")], names=["i0", "i1"])
    out = schema.validate(pd.DataFrame({"a": [1, 2]}, index=mi))
    assert isinstance(out.index, pd.MultiIndex)

    mi_bad = pd.MultiIndex.from_tuples(
        [(0, "x"), (1, "z")], names=["i0", "i1"]
    )
    with pytest.raises(pa.errors.SchemaErrors):
        schema.validate(pd.DataFrame({"a": [1, 2]}, index=mi_bad), lazy=True)


def test_column_coerce(df):
    """Column-level coerce=True coerces dtypes for pandas frames (delegated to
    the native pandas backend), without emitting a no-op warning."""
    schema = pa.DataFrameSchema({"x": pa.Column(float, coerce=True)})
    with warnings.catch_warnings():
        warnings.simplefilter("error", SchemaWarning)
        out = schema.validate(df)
    assert out["x"].dtype == "float64"
    assert out["x"].tolist() == [1.0, 2.0, 3.0]


def test_column_coerce_failure():
    schema = pa.DataFrameSchema({"x": pa.Column(int, coerce=True)})
    bad = pd.DataFrame({"x": ["a", "2"]})
    with pytest.raises(pa.errors.SchemaErrors):
        schema.validate(bad, lazy=True)


def test_schema_level_coerce():
    schema = pa.DataFrameSchema(
        {"a": pa.Column(float), "b": pa.Column(str)}, coerce=True
    )
    out = schema.validate(pd.DataFrame({"a": [1, 2], "b": [10, 20]}))
    assert out["a"].dtype == "float64"
    # str coercion yields object on pandas < 3 and pd.StringDtype on
    # pandas >= 3; derive the expected dtype from the engine so the
    # assertion is version-agnostic (matches the native pandas backend).
    from pandera.engines import pandas_engine

    expected_b_dtype = (
        pandas_engine.Engine.dtype(str).coerce(pd.Series([10, 20])).dtype
    )
    assert out["b"].dtype == expected_b_dtype


def test_coerce_numpy_dtype_via_narwhals_cast():
    """Plain numpy dtypes are coerced Narwhals-native via ``nw.cast``."""
    schema = pa.DataFrameSchema({"a": pa.Column("int64", coerce=True)})
    out = schema.validate(pd.DataFrame({"a": ["1", "2", "3"]}))
    assert out["a"].dtype == "int64"
    assert out["a"].tolist() == [1, 2, 3]


def test_coerce_nullable_int_falls_back_to_pandas_engine():
    """Nullable ``Int64`` coercion falls back to the pandas dtype engine so
    ``<NA>`` values are preserved (Narwhals cast would drop the nullability).
    """
    schema = pa.DataFrameSchema(
        {"a": pa.Column("Int64", coerce=True, nullable=True)}
    )
    out = schema.validate(pd.DataFrame({"a": [1.0, None, 3.0]}))
    assert str(out["a"].dtype) == "Int64"
    assert out["a"].isna().tolist() == [False, True, False]


def test_coerce_categorical_and_tz_datetime_fidelity():
    """Extension dtypes (category, tz-aware datetime) coerce with native
    pandas fidelity via the pandas dtype engine fallback."""
    cat = pa.DataFrameSchema({"a": pa.Column("category", coerce=True)})
    assert str(
        cat.validate(pd.DataFrame({"a": ["x", "y", "x"]}))["a"].dtype
    ) == ("category")

    tz = pa.DataFrameSchema(
        {"a": pa.Column("datetime64[ns, UTC]", coerce=True)}
    )
    out = tz.validate(
        pd.DataFrame({"a": pd.to_datetime(["2021-01-01", "2021-01-02"])})
    )
    assert str(out["a"].dtype) == "datetime64[ns, UTC]"


def test_coerce_failure_reports_offending_values():
    """A failed coercion reports the offending value (pandas-engine fallback)."""
    schema = pa.DataFrameSchema({"a": pa.Column(int, coerce=True)})
    with pytest.raises(pa.errors.SchemaErrors) as exc_info:
        schema.validate(pd.DataFrame({"a": ["x", "2"]}), lazy=True)
    coerce_fc = exc_info.value.failure_cases.query(
        "check == \"coerce_dtype('int64')\""
    )
    assert "x" in coerce_fc["failure_case"].astype(str).str.cat()


def test_index_coerce():
    schema = pa.DataFrameSchema(
        {"a": pa.Column(int)},
        index=pa.Index(int, coerce=True, name="idx"),
    )
    out = schema.validate(
        pd.DataFrame({"a": [1, 2]}, index=pd.Index(["0", "1"], name="idx"))
    )
    assert out.index.dtype == "int64"


def test_index_coerce_failure():
    schema = pa.DataFrameSchema(
        {"a": pa.Column(int)},
        index=pa.Index(int, coerce=True, name="idx"),
    )
    bad = pd.DataFrame({"a": [1, 2]}, index=pd.Index(["x", "y"], name="idx"))
    with pytest.raises((SchemaError, SchemaErrors)):
        schema.validate(bad, lazy=True)


def test_multiindex_coerce():
    schema = pa.DataFrameSchema(
        {"a": pa.Column(int)},
        index=pa.MultiIndex(
            [
                pa.Index(int, coerce=True, name="i0"),
                pa.Index(str, name="i1"),
            ]
        ),
    )
    mi = pd.MultiIndex.from_tuples(
        [("0", "x"), ("1", "y")], names=["i0", "i1"]
    )
    out = schema.validate(pd.DataFrame({"a": [1, 2]}, index=mi))
    assert out.index.get_level_values("i0").dtype == "int64"


def test_custom_parsers():
    """Custom parsers= run on the native frame inside the narwhals backend."""
    schema = pa.DataFrameSchema(
        {"a": pa.Column(int, pa.Check.ge(0))},
        parsers=pa.Parser(lambda d: d.assign(a=d["a"].abs())),
    )
    # The check ge(0) would fail on the raw negative data; it passes only
    # because the parser ran first.
    out = schema.validate(pd.DataFrame({"a": [-1, -2, 3]}))
    assert out["a"].tolist() == [1, 2, 3]


def test_custom_parsers_chained():
    schema = pa.DataFrameSchema(
        {"a": pa.Column(int)},
        parsers=[
            pa.Parser(lambda d: d.assign(a=d["a"] + 1)),
            pa.Parser(lambda d: d.assign(a=d["a"] * 2)),
        ],
    )
    out = schema.validate(pd.DataFrame({"a": [1, 2]}))
    assert out["a"].tolist() == [4, 6]  # (x+1)*2


def test_custom_parsers_element_wise():
    schema = pa.DataFrameSchema(
        {"a": pa.Column(int), "b": pa.Column(int)},
        parsers=pa.Parser(lambda row: row * 2, element_wise=True),
    )
    out = schema.validate(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))
    assert out["a"].tolist() == [2, 4]
    assert out["b"].tolist() == [6, 8]


def test_add_missing_columns():
    schema = pa.DataFrameSchema(
        {"a": pa.Column(int), "b": pa.Column(float, default=1.5)},
        add_missing_columns=True,
    )
    out = schema.validate(pd.DataFrame({"a": [1, 2, 3]}))
    assert out["b"].tolist() == [1.5, 1.5, 1.5]
    assert out["b"].dtype == "float64"
    assert list(out.columns) == ["a", "b"]


def test_add_missing_columns_ordering_and_multiple():
    """Missing columns are inserted at their schema position, not appended."""
    schema = pa.DataFrameSchema(
        {
            "a": pa.Column(int, default=0),
            "b": pa.Column(int),
            "c": pa.Column(int, default=0),
        },
        add_missing_columns=True,
    )
    out = schema.validate(pd.DataFrame({"b": [1]}))
    assert list(out.columns) == ["a", "b", "c"]


def test_add_missing_columns_extension_dtype_fidelity():
    """Added extension-dtype columns get the correct native dtype (not numpy):
    nullable Int64 with a default, and nullable-no-default (all <NA>)."""
    schema = pa.DataFrameSchema(
        {
            "a": pa.Column(int),
            "b": pa.Column("Int64", default=5, nullable=True),
            "c": pa.Column("Int64", nullable=True),
        },
        add_missing_columns=True,
    )
    out = schema.validate(pd.DataFrame({"a": [1, 2]}))
    assert str(out["b"].dtype) == "Int64"
    assert out["b"].tolist() == [5, 5]
    assert str(out["c"].dtype) == "Int64"
    assert out["c"].isna().all()


def test_add_missing_columns_requires_default():
    schema = pa.DataFrameSchema(
        {"a": pa.Column(int), "b": pa.Column(int)},
        add_missing_columns=True,
    )
    with pytest.raises(pa.errors.SchemaErrors):
        schema.validate(pd.DataFrame({"a": [1]}), lazy=True)


def test_set_defaults():
    schema = pa.DataFrameSchema(
        {"a": pa.Column(float, default=0.0, nullable=False)}
    )
    out = schema.validate(pd.DataFrame({"a": [1.0, None, 3.0]}))
    # only the null value is replaced by the default; others are untouched.
    assert out["a"].tolist() == [1.0, 0.0, 3.0]


def test_set_defaults_no_default_is_noop():
    schema = pa.DataFrameSchema({"a": pa.Column(float, nullable=True)})
    out = schema.validate(pd.DataFrame({"a": [1.0, None, 3.0]}))
    assert out["a"].isna().tolist() == [False, True, False]


def test_unique_column_names():
    schema = pa.DataFrameSchema(
        {"a": pa.Column(int)}, unique_column_names=True
    )
    dup = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "a"])
    with pytest.raises(SchemaError, match="multiple columns with label"):
        schema.validate(dup)
    with pytest.raises(SchemaErrors) as exc_info:
        schema.validate(dup, lazy=True)
    assert (
        exc_info.value.schema_errors[0].reason_code
        == pa.errors.SchemaErrorReason.DUPLICATE_COLUMN_LABELS
    )


def test_groupby_check():
    """Column check groups (groupby=) run Narwhals-native via apply_groupby."""
    schema = pa.DataFrameSchema(
        {
            "height": pa.Column(
                float,
                pa.Check(
                    lambda g: g["M"].mean() > g["F"].mean(), groupby="sex"
                ),
            ),
            "sex": pa.Column(str),
        }
    )
    ok = pd.DataFrame(
        {"height": [6.0, 5.9, 5.4, 5.5], "sex": ["M", "M", "F", "F"]}
    )
    schema.validate(ok)

    bad = pd.DataFrame(
        {"height": [5.0, 5.1, 6.4, 6.5], "sex": ["M", "M", "F", "F"]}
    )
    with pytest.raises(SchemaErrors):
        schema.validate(bad, lazy=True)


def test_groupby_check_returns_dict():
    """A groupby check returning a per-group dict of bools is reduced to
    pass only when every group passes."""
    schema = pa.DataFrameSchema(
        {
            "val": pa.Column(
                int,
                pa.Check(
                    lambda g: {k: (v > 0).all() for k, v in g.items()},
                    groupby="grp",
                ),
            ),
            "grp": pa.Column(str),
        }
    )
    ok = pd.DataFrame({"val": [1, 2, 3, 4], "grp": ["a", "a", "b", "b"]})
    schema.validate(ok)

    bad = pd.DataFrame({"val": [1, 2, -3, 4], "grp": ["a", "a", "b", "b"]})
    with pytest.raises(SchemaErrors):
        schema.validate(bad, lazy=True)


def test_groupby_check_with_groups_filter():
    """The groups= kwarg restricts which groups the check runs on."""
    schema = pa.DataFrameSchema(
        {
            "val": pa.Column(
                int,
                pa.Check(
                    lambda g: g["a"].mean() > 0, groupby="grp", groups=["a"]
                ),
            ),
            "grp": pa.Column(str),
        }
    )
    # group "b" is negative but excluded by groups=["a"], so validation passes.
    df = pd.DataFrame({"val": [1, 2, -5, -6], "grp": ["a", "a", "b", "b"]})
    schema.validate(df)


def test_hypothesis_check():
    """Hypothesis checks run through the native pandas hypothesis backend."""
    pytest.importorskip("scipy")
    schema = pa.DataFrameSchema(
        {
            "height": pa.Column(
                float,
                pa.Hypothesis.two_sample_ttest(
                    sample1="M",
                    sample2="F",
                    groupby="sex",
                    relationship="greater_than",
                    alpha=0.05,
                ),
            ),
            "sex": pa.Column(str),
        }
    )
    ok = pd.DataFrame(
        {
            "height": [7.0, 6.9, 6.8, 5.4, 5.5, 5.3],
            "sex": ["M", "M", "M", "F", "F", "F"],
        }
    )
    schema.validate(ok)

    bad = pd.DataFrame(
        {
            "height": [5.0, 5.1, 5.2, 6.4, 6.5, 6.6],
            "sex": ["M", "M", "M", "F", "F", "F"],
        }
    )
    with pytest.raises(SchemaErrors):
        schema.validate(bad, lazy=True)


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


def test_scalar_failure_cases_without_polars(df, monkeypatch):
    """Scalar failure cases build pandas frames on polars-free installs."""
    import pandera.backends.narwhals.base as narwhals_base

    monkeypatch.setattr(narwhals_base, "pl", None)
    schema = pa.DataFrameSchema({"x": pa.Column(str)})
    with pytest.raises(SchemaErrors) as exc_info:
        schema.validate(df, lazy=True)
    failure_cases = exc_info.value.failure_cases
    assert isinstance(failure_cases, pd.DataFrame)
    assert failure_cases["failure_case"].tolist() == ["int64"]


def test_failure_cases_metadata_with_non_frame_error_data(df):
    """Non-frame err.data falls back to schema-module implementation detection."""
    from pandera.errors import SchemaErrorReason

    schema = pa.DataFrameSchema({"x": pa.Column(int)})
    backend = schema.get_backend(df)
    err = SchemaError(
        schema=pa.Column(int, name="x"),
        data="not-a-frame",
        message="boom",
        failure_cases="int64",
        reason_code=SchemaErrorReason.WRONG_DATATYPE,
    )
    metadata = backend.failure_cases_metadata("schema", [err])
    assert isinstance(metadata.failure_cases, pd.DataFrame)


def test_check_dtype_iterable_result(df):
    """Iterable dtype-check results are AND-reduced (native pandas semantics)."""
    import narwhals.stable.v1 as nw

    from pandera.backends.narwhals.components import ColumnBackend
    from pandera.engines import numpy_engine

    class IterableResultInt64(numpy_engine.Int64):
        def check(self, pandera_dtype, data_container=None):
            return iter([True, True])

    column = pa.Column(int, name="x")
    column._dtype = IterableResultInt64()
    lf = nw.from_native(df).lazy()
    results = ColumnBackend().check_dtype(lf, column)
    assert all(result.passed for result in results)


def test_is_pandas_like_helper(df):
    """_is_pandas_like distinguishes narwhals pandas frames from other values."""
    import narwhals.stable.v1 as nw

    from pandera.api.narwhals.utils import _is_pandas_like

    assert _is_pandas_like(nw.from_native(df))
    assert _is_pandas_like(nw.from_native(df).lazy())
    assert not _is_pandas_like("not a frame")
    assert not _is_pandas_like(df)


def test_pandas_column_selector_property():
    """The pandas Column exposes the selector property used by the backend."""
    assert pa.Column(int, name="a").selector == "a"
    assert pa.Column(int, name="a", regex=True).selector == "^a$"
    assert pa.Column(int, name="^a.*$", regex=True).selector == "^a.*$"
    assert pa.Column(int, name=("a", "b")).selector == ("a", "b")
