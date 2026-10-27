"""Failure-case reporting limits must not limit invalid-row removal."""

import pandas as pd
import pytest

import pandera.pandas as pa


@pytest.mark.parametrize("n_failure_cases", [0, 1, None])
@pytest.mark.parametrize("element_wise", [False, True])
@pytest.mark.parametrize(
    "values", [[], [-1.0, -2.0], [-1.0, 2.0, -3.0], [1.0, 2.0]]
)
@pytest.mark.parametrize(
    "schema_type", ["dataframe", "series", "column", "nested"]
)
def test_drop_invalid_rows_with_failure_limit(
    n_failure_cases, element_wise, values, schema_type
):
    """Column and series checks remove every failing row regardless of cap."""
    frame = pd.DataFrame({"x": pd.Series(values, dtype=float)})
    check = pa.Check(
        lambda x: x > 0,
        element_wise=element_wise,
        n_failure_cases=n_failure_cases,
    )
    expected = frame.loc[frame["x"] > 0]
    data = frame
    if schema_type == "dataframe":
        schema = pa.DataFrameSchema(
            {"x": pa.Column(float, check)}, drop_invalid_rows=True
        )
    elif schema_type == "series":
        schema = pa.SeriesSchema(float, check, drop_invalid_rows=True)
        data, expected = frame["x"], expected["x"]
    elif schema_type == "column":
        schema = pa.Column(float, check, name="x", drop_invalid_rows=True)
    else:
        schema = pa.DataFrameSchema(
            {"x": pa.Column(float, check, drop_invalid_rows=True)}
        )

    validated = schema.validate(data, lazy=True)

    if schema_type == "series":
        pd.testing.assert_series_equal(validated, expected)
    else:
        pd.testing.assert_frame_equal(validated, expected)


@pytest.mark.parametrize("n_failure_cases", [0, 1, None])
@pytest.mark.parametrize(
    "output_type", ["series", "dataframe", "element_wise"]
)
@pytest.mark.parametrize("values", [[], [-1, -2], [-1, 2, -3], [1, 2]])
def test_drop_invalid_rows_dataframe_check_with_failure_limit(
    n_failure_cases, output_type, values
):
    """Dataframe check masks remain complete when diagnostics are truncated."""
    frame = pd.DataFrame({"x": pd.Series(values, dtype="int64")})
    check_fn = (
        (lambda df: df > 0)
        if output_type == "dataframe"
        else (lambda df: df["x"] > 0)
    )
    schema = pa.DataFrameSchema(
        {"x": pa.Column(int)},
        checks=pa.Check(
            check_fn,
            element_wise=output_type == "element_wise",
            n_failure_cases=n_failure_cases,
        ),
        drop_invalid_rows=True,
    )

    validated = schema.validate(frame, lazy=True)

    pd.testing.assert_frame_equal(validated, frame.loc[frame["x"] > 0])


@pytest.mark.parametrize("n_failure_cases", [0, 1, None])
def test_drop_invalid_rows_with_failure_limit_across_columns(n_failure_cases):
    """A dataframe mask removes rows failing in any checked column."""
    frame = pd.DataFrame(
        {"x": [-1, 2, 3], "y": [4, -5, 6]}, index=["first", "second", "third"]
    )
    schema = pa.DataFrameSchema(
        {"x": pa.Column(int), "y": pa.Column(int)},
        checks=pa.Check(lambda df: df > 0, n_failure_cases=n_failure_cases),
        drop_invalid_rows=True,
    )

    validated = schema.validate(frame, lazy=True)

    pd.testing.assert_frame_equal(validated, frame.loc[["third"]])


@pytest.mark.parametrize("n_failure_cases", [0, 1, None])
def test_failure_limit_still_caps_diagnostics(n_failure_cases):
    """Using the full row mask for dropping does not expand error reports."""
    frame = pd.DataFrame({"x": [-1, -2, 3]})
    schema = pa.DataFrameSchema(
        {"x": pa.Column(int, pa.Check.gt(0, n_failure_cases=n_failure_cases))}
    )

    with pytest.raises(pa.errors.SchemaErrors) as exc_info:
        schema.validate(frame, lazy=True)

    assert len(exc_info.value.failure_cases) == (
        2 if n_failure_cases is None else n_failure_cases
    )


@pytest.mark.parametrize("n_failure_cases", [0, 1, None])
def test_drop_invalid_rows_with_failure_limit_requires_lazy(n_failure_cases):
    """The reporting cap does not change the lazy-validation requirement."""
    schema = pa.DataFrameSchema(
        {"x": pa.Column(int, pa.Check.gt(0, n_failure_cases=n_failure_cases))},
        drop_invalid_rows=True,
    )

    with pytest.raises(pa.errors.SchemaDefinitionError, match="lazy"):
        schema.validate(pd.DataFrame({"x": [-1, 2]}))


@pytest.mark.parametrize("n_failure_cases", [0, 1, None])
@pytest.mark.parametrize("ignore_na", [False, True])
@pytest.mark.parametrize("output_type", ["column", "dataframe"])
def test_drop_invalid_rows_failure_limit_preserves_ignored_nulls(
    n_failure_cases, ignore_na, output_type
):
    """Only rows skipped by ignore_na survive a failed check."""
    frame = pd.DataFrame({"x": [-1.0, None, -2.0, 3.0]})

    def make_schema(limit):
        check = pa.Check(
            lambda obj: obj > 0, n_failure_cases=limit, ignore_na=ignore_na
        )
        if output_type == "dataframe":
            return pa.DataFrameSchema(
                {"x": pa.Column(float, nullable=True)},
                checks=check,
                drop_invalid_rows=True,
            )
        return pa.DataFrameSchema(
            {"x": pa.Column(float, check, nullable=True)},
            drop_invalid_rows=True,
        )

    schema = make_schema(n_failure_cases)

    validated = schema.validate(frame, lazy=True)

    expected = frame.loc[[1, 3]] if ignore_na else frame.loc[[3]]
    pd.testing.assert_frame_equal(validated, expected)


@pytest.mark.parametrize("n_failure_cases", [0, 1, None])
def test_drop_invalid_rows_failure_limit_keeps_warning_rows(n_failure_cases):
    """A warning check still does not cause rows to be removed."""
    frame = pd.DataFrame({"x": [-1, -2, 3]})
    schema = pa.DataFrameSchema(
        {
            "x": pa.Column(
                int,
                pa.Check.gt(
                    0, n_failure_cases=n_failure_cases, raise_warning=True
                ),
            )
        },
        drop_invalid_rows=True,
    )

    with pytest.warns(pa.errors.SchemaWarning):
        validated = schema.validate(frame, lazy=True)

    pd.testing.assert_frame_equal(validated, frame)


@pytest.mark.parametrize("n_failure_cases", [0, 1, None])
@pytest.mark.parametrize(
    "index",
    [
        pd.Index(["bad1", "bad2", "good"], name="row"),
        pd.date_range("2026-01-01", periods=3, name="row"),
        pd.MultiIndex.from_tuples(
            [("a", 1), ("a", 2), ("b", 1)], names=["group", "row"]
        ),
    ],
)
def test_drop_invalid_rows_failure_limit_preserves_index(
    n_failure_cases, index
):
    """Full masks preserve each supported unique index's type and names."""
    frame = pd.DataFrame({"x": [-1, -2, 3]}, index=index)
    schema = pa.DataFrameSchema(
        {"x": pa.Column(int, pa.Check.gt(0, n_failure_cases=n_failure_cases))},
        drop_invalid_rows=True,
    )

    validated = schema.validate(frame, lazy=True)

    pd.testing.assert_frame_equal(validated, frame.iloc[[2]])


@pytest.mark.parametrize("n_failure_cases", [0, 1, None])
def test_drop_invalid_rows_failure_limit_multiple_checks(n_failure_cases):
    """Every limited check contributes its full set of failing rows."""
    frame = pd.DataFrame({"x": [-1, -2, 1, 4, 5]})
    schema = pa.DataFrameSchema(
        {
            "x": pa.Column(
                int,
                [
                    pa.Check.gt(0, n_failure_cases=n_failure_cases),
                    pa.Check.lt(3, n_failure_cases=n_failure_cases),
                ],
            )
        },
        drop_invalid_rows=True,
    )

    validated = schema.validate(frame, lazy=True)

    pd.testing.assert_frame_equal(validated, frame.iloc[[2]])


@pytest.mark.parametrize("n_failure_cases", [0, 1, None])
def test_drop_invalid_rows_failure_limit_respects_subsample(n_failure_cases):
    """A reporting cap does not extend validation to unsampled rows."""
    frame = pd.DataFrame({"x": [-1, -2, -3, 4]})
    schema = pa.DataFrameSchema(
        {"x": pa.Column(int, pa.Check.gt(0, n_failure_cases=n_failure_cases))},
        drop_invalid_rows=True,
    )

    validated = schema.validate(frame, head=2, lazy=True)

    pd.testing.assert_frame_equal(validated, frame.iloc[2:])
