"""Tests the way Columns are Checked"""

import copy

import pandas as pd
import pytest

from pandera.pandas import (
    Bool,
    Check,
    Column,
    DataFrameSchema,
    Float,
    Index,
    Int,
    SeriesSchema,
    String,
    errors,
)
from pandera.backends.pandas import error_formatters


def test_vectorized_checks() -> None:
    """Test that using element-wise checking returns and errors as expected."""
    schema = SeriesSchema(
        Int, Check(lambda s: s.value_counts() == 2, element_wise=False)
    )
    validated_series = schema.validate(pd.Series([1, 1, 2, 2, 3, 3]))
    assert isinstance(validated_series, pd.Series)

    # error case
    with pytest.raises(errors.SchemaError):
        schema.validate(pd.Series([1, 2, 3]))


def test_check_groupby() -> None:
    """Tests uses of groupby to specify dependencies between one column and a
    single other column, including error handling."""
    schema = DataFrameSchema(
        columns={
            "col1": Column(
                Int,
                [
                    Check(lambda s: s["foo"] > 10, groupby="col2"),
                    Check(lambda s: s["bar"] < 10, groupby=["col2"]),
                    Check(
                        lambda s: s["foo"] > 10,
                        groupby=lambda df: df.groupby("col2"),
                    ),
                    Check(
                        lambda s: s["bar"] < 10,
                        groupby=lambda df: df.groupby("col2"),
                    ),
                ],
            ),
            "col2": Column(String, Check(lambda s: s.isin(["foo", "bar"]))),
        },
        index=Index(Int, name="data_id"),
    )

    df_pass = pd.DataFrame(
        data={
            "col1": [7, 8, 9, 11, 12, 13],
            "col2": ["bar", "bar", "bar", "foo", "foo", "foo"],
        },
        index=pd.Series([1, 2, 3, 4, 5, 6], name="data_id"),
    )

    df = schema.validate(df_pass)
    assert isinstance(df, pd.DataFrame)
    assert len(df.columns) == 2
    assert set(df.columns) == {"col1", "col2"}

    # raise errors.SchemaError when Check fails
    df_fail_on_bar = pd.DataFrame(
        data={
            "col1": [7, 8, 20, 11, 12, 13],
            "col2": ["bar", "bar", "bar", "foo", "foo", "foo"],
        },
        index=pd.Series([1, 2, 3, 4, 5, 6], name="data_id"),
    )
    df_fail_on_foo = pd.DataFrame(
        data={
            "col1": [7, 8, 9, 11, 1, 13],
            "col2": ["bar", "bar", "bar", "foo", "foo", "foo"],
        },
        index=pd.Series([1, 2, 3, 4, 5, 6], name="data_id"),
    )
    # raise errors.SchemaError when groupby column doesn't exist
    df_fail_no_column = pd.DataFrame(
        data={
            "col1": [7, 8, 20, 11, 12, 13],
        },
        index=pd.Series([1, 2, 3, 4, 5, 6], name="data_id"),
    )

    for df in [df_fail_on_bar, df_fail_on_foo, df_fail_no_column]:
        with pytest.raises(errors.SchemaError):
            schema.validate(df)


def test_check_groupby_multiple_columns() -> None:
    """Tests uses of groupby to specify dependencies between one column and a
    number of other columns, including error handling."""
    schema = DataFrameSchema(
        {
            "col1": Column(
                Int,
                [
                    Check(
                        lambda s: s[("bar", True)].sum() == 16,  # 7 + 9
                        groupby=["col2", "col3"],
                    ),
                ],
            ),
            "col2": Column(String, Check(lambda s: s.isin(["foo", "bar"]))),
            "col3": Column(Bool),
        }
    )

    df_pass = pd.DataFrame(
        {
            "col1": [7, 8, 9, 11, 12, 13],
            "col2": ["bar", "bar", "bar", "foo", "foo", "foo"],
            "col3": [True, False, True, False, True, False],
        }
    )

    df = schema.validate(df_pass)
    assert isinstance(df, pd.DataFrame)
    assert len(df.columns) == 3
    assert set(df.columns) == {"col1", "col2", "col3"}


def test_check_groups() -> None:
    """Tests uses of groupby and groups (for values within columns)."""
    schema = DataFrameSchema(
        {
            "col1": Column(
                Int,
                [
                    Check(
                        lambda s: s["foo"] > 10, groupby="col2", groups=["foo"]
                    ),
                    Check(
                        lambda s: s["foo"] > 10, groupby="col2", groups="foo"
                    ),
                ],
            ),
            "col2": Column(String, Check(lambda s: s.isin(["foo", "bar"]))),
        }
    )

    df = pd.DataFrame(
        {
            "col1": [7, 8, 9, 11, 12, 13],
            "col2": ["bar", "bar", "bar", "foo", "foo", "foo"],
        }
    )

    validated_df = schema.validate(df)
    assert isinstance(validated_df, pd.DataFrame)
    assert len(validated_df.columns) == 2
    assert set(validated_df.columns) == {"col1", "col2"}

    # raise KeyError when groups does not include a particular group name
    schema_fail_key_error = DataFrameSchema(
        {
            "col1": Column(
                Int,
                [
                    Check(
                        lambda s: s["bar"] > 10, groupby="col2", groups="foo"
                    ),
                ],
            ),
            "col2": Column(String, Check(lambda s: s.isin(["foo", "bar"]))),
        }
    )
    with pytest.raises(
        errors.SchemaError,
        match=r'Error while executing check function: KeyError\("bar"\)',
    ):
        schema_fail_key_error.validate(df)

    # raise KeyError when the group does not exist in the groupby column when
    # referenced in the Check function
    schema_fail_nonexistent_key_in_fn = DataFrameSchema(
        {
            "col1": Column(
                Int,
                [
                    Check(
                        lambda s: s["baz"] > 10, groupby="col2", groups=["foo"]
                    ),
                ],
            ),
            "col2": Column(String, Check(lambda s: s.isin(["foo", "bar"]))),
        }
    )
    with pytest.raises(
        errors.SchemaError,
        match=r'Error while executing check function: KeyError\("baz"\)',
    ):
        schema_fail_nonexistent_key_in_fn.validate(df)

    # raise KeyError when the group does not exist in the groups argument.
    schema_fail_nonexistent_key_in_groups = DataFrameSchema(
        {
            "col1": Column(
                Int,
                [
                    Check(
                        lambda s: s["foo"] > 10, groupby="col2", groups=["baz"]
                    ),
                ],
            ),
            "col2": Column(String, Check(lambda s: s.isin(["foo", "bar"]))),
        }
    )
    with pytest.raises(errors.SchemaError):
        schema_fail_nonexistent_key_in_groups.validate(df)


def test_groupby_init_exceptions() -> None:
    """Test that when using a groupby it errors properly across a variety of
    API-specific differences."""

    def init_schema_element_wise():
        DataFrameSchema(
            {
                "col1": Column(
                    Int,
                    [
                        Check(
                            lambda s: s["foo"] > 10,
                            element_wise=True,
                            groupby=["col2"],
                        ),
                    ],
                ),
                "col2": Column(
                    String, Check(lambda s: s.isin(["foo", "bar"]))
                ),
            }
        )

    # can't use groupby in Checks where element_wise == True
    with pytest.raises(
        errors.SchemaInitError,
        match=r"^Cannot use groupby when element_wise=True.",
    ):
        init_schema_element_wise()

    # raise errors.SchemaInitError even when the schema doesn't specify column
    # key for groupby column
    def init_schema_no_groupby_column():
        DataFrameSchema(
            {
                "col1": Column(
                    Int,
                    [
                        Check(lambda s: s["foo"] > 10, groupby=["col2"]),
                    ],
                ),
            }
        )

    with pytest.raises(errors.SchemaInitError):
        init_schema_no_groupby_column()

    # can't use groupby argument in SeriesSchema or Index objects
    for schema_class in [SeriesSchema, Index]:
        with pytest.raises(
            errors.SchemaInitError, match="^Cannot use groupby checks with"
        ):
            schema_class(Int, Check(lambda s: s["bar"] == 1, groupby="foo"))


def test_dataframe_checks() -> None:
    """Tests that dataframe checks validate, error when a DataFrame doesn't
    comply with the schema, simple tests of the groupby checks which are
    covered in more detail above."""
    schema = DataFrameSchema(
        columns={
            "col1": Column(Int),
            "col2": Column(Float),
            "col3": Column(String),
            "col4": Column(String),
        },
        checks=[
            Check(lambda df: df["col1"] < df["col2"]),
            Check(lambda df: df["col3"] == df["col4"]),
        ],
    )
    df = pd.DataFrame(
        {
            "col1": [1, 2, 3],
            "col2": [2.0, 3.0, 4.0],
            "col3": ["foo", "bar", "baz"],
            "col4": ["foo", "bar", "baz"],
        }
    )

    assert isinstance(schema.validate(df), pd.DataFrame)

    # test invalid schema error raising
    invalid_df = df.copy()
    invalid_df["col1"] = invalid_df["col1"] * 3

    with pytest.raises(errors.SchemaError):
        schema.validate(invalid_df)

    # test groupby checks
    groupby_check_schema = DataFrameSchema(
        columns={
            "col1": Column(Int),
            "col3": Column(String),
        },
        checks=[
            Check(lambda g: g["foo"]["col1"].iat[0] == 1, groupby="col3"),
            Check(lambda g: g["foo"]["col2"].iat[0] == 2.0, groupby="col3"),
            Check(lambda g: g["foo"]["col3"].iat[0] == "foo", groupby="col3"),
            Check(
                lambda g: g[("foo", "foo")]["col1"].iat[0] == 1,
                groupby=["col3", "col4"],
            ),
        ],
    )
    assert isinstance(groupby_check_schema.validate(df), pd.DataFrame)

    # test element-wise checks
    element_wise_check_schema = DataFrameSchema(
        columns={
            "col1": Column(Int),
            "col2": Column(Float),
        },
        checks=Check(lambda row: row["col1"] < row["col2"], element_wise=True),
    )
    assert isinstance(element_wise_check_schema.validate(df), pd.DataFrame)


def test_reshape_failure_cases_exceptions() -> None:
    """Tests that the reshape_failure_cases method correctly produces a
    TypeError."""
    # pylint: disable=W0212, E1121
    # disabling pylint because this function should be private to the class and
    # it's ok to access it because the function needs to be tested.
    check = Check(lambda x: x.isna().sum() == 0)
    for data in [1, "foobar", 1.0, {"key": "value"}, list(range(10))]:
        with pytest.raises(TypeError):
            error_formatters.reshape_failure_cases(
                data, bool(check.n_failure_cases)  # type: ignore
            )


def test_check_equality_operators() -> None:
    """Test the usage of == between a Check and an entirely different Check,
    and a non-Check."""
    check = Check(lambda g: g["foo"]["col1"].iat[0] == 1, groupby="col3")

    not_equal_check = Check(lambda x: x.isna().sum() == 0)
    assert check == copy.deepcopy(check)
    assert check != not_equal_check
    assert check != "not a check"


def test_equality_operators_functional_equivalence() -> None:
    """Test the usage of == for Checks where the Check callable object has
    the same implementation."""
    main_check = Check(lambda g: g["foo"]["col1"].iat[0] == 1, groupby="col3")
    same_check = Check(lambda h: h["foo"]["col1"].iat[0] == 1, groupby="col3")

    assert main_check == same_check


def test_raise_warning_series() -> None:
    """Test that checks with raise_warning=True raise a warning."""
    data = pd.Series([-1, -2, -3])
    error_schema = SeriesSchema(checks=Check(lambda s: s > 0))
    warning_schema = SeriesSchema(
        checks=Check(lambda s: s > 0, raise_warning=True)
    )

    with pytest.raises(errors.SchemaError):
        error_schema(data)

    with pytest.warns(errors.SchemaWarning):
        warning_schema(data)

    # For compatibility with old behaviour of raise_warning to give UserWarning
    with pytest.warns(UserWarning):
        warning_schema(data)


def test_raise_warning_dataframe() -> None:
    """Test that checks with raise_warning=True raise a warning."""
    data = pd.DataFrame({"positive_numbers": [-1, -2, -3]})
    error_schema = DataFrameSchema(
        {
            "positive_numbers": Column(checks=Check(lambda s: s > 0)),
        }
    )
    warning_schema = DataFrameSchema(
        {
            "positive_numbers": Column(
                checks=Check(lambda s: s > 0, raise_warning=True)
            ),
        }
    )

    with pytest.raises(errors.SchemaError):
        error_schema(data)

    with pytest.warns(errors.SchemaWarning):
        warning_schema(data)


def test_dataframe_schema_check() -> None:
    """Test that DataFrameSchema-level Checks work properly."""
    data = pd.DataFrame([range(10) for _ in range(10)])

    schema_check_return_bool = DataFrameSchema(
        checks=Check(lambda df: (df < 10).all())
    )
    assert isinstance(schema_check_return_bool.validate(data), pd.DataFrame)

    schema_check_return_series = DataFrameSchema(
        checks=Check(lambda df: df[0] < 10)
    )
    assert isinstance(schema_check_return_series.validate(data), pd.DataFrame)

    schema_check_return_df = DataFrameSchema(checks=Check(lambda df: df < 10))
    assert isinstance(schema_check_return_df.validate(data), pd.DataFrame)


def test_dataframe_check_schema_error() -> None:
    """Test that DataFramSchema-level checks raises errors."""

    schema = DataFrameSchema(
        checks=Check(
            lambda df: df["a"].isna() | ~df["b"].isna(), ignore_na=False
        )
    )
    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 1],
            "b": [1, 2, None, None],
        }
    )

    try:
        schema(df, lazy=True)
    except errors.SchemaErrors as exc:
        assert pd.isna(
            exc.failure_cases.query(
                "index == 2 & column == 'b'"
            ).failure_case.iloc[0]
        )
        assert pd.isna(
            exc.failure_cases.query(
                "index == 3 & column == 'b'"
            ).failure_case.iloc[0]
        )


def test_prepare_series_check_output_df_level():
    """Test that dataframe-level checks only ignore rows where all values are null."""
    df = pd.DataFrame(
        {
            "a": [1, 1, 2, 2, 3, 3, None],
            "b": [2, 1, 4, 3, 6, 5, None],
            "c": [None] * 7,
        }
    )
    check = Check(lambda df: df["b"] == df["a"] * 2, ignore_na=True)
    # The last record should evaluate to True, since all values are null
    expected_output = [True, False, True, False, True, False, True]
    result = check(df)
    assert result.check_output.tolist() == expected_output


# pylint: disable=unused-argument
def test_custom_check_error_is_failure_case(extra_registered_checks):
    """Test that an error in a custom check is returned as a failure case"""
    test_schema = DataFrameSchema(checks=[Check.raise_an_error_check()])

    df = pd.DataFrame()

    try:
        test_schema.validate(df, lazy=True)
    except errors.SchemaErrors as err:
        assert err.error_counts == {"CHECK_ERROR": 1}
        assert (
            err.message["DATA"]["CHECK_ERROR"][0]["check"]
            == "raise_an_error_check"
        )


def test_check_backend_not_found():
    """Test that checks complain if a backend is not register for that type."""

    class CustomDataObject:
        """Custom data object."""

    dummy_check = Check(lambda _: True)

    with pytest.raises(KeyError, match="Backend not found for class"):
        dummy_check(CustomDataObject())


def test_check_output_dtype_with_empty_datetime():
    from pandera.backends.pandas.register import register_pandas_backends

    # NOTE: this should automatically be handles in the check.__call__ method
    register_pandas_backends("pandas.DataFrame")

    check = Check(lambda _: True, element_wise=True)
    df = pd.DataFrame({"year_mon": pd.Series(dtype="datetime64[D]")})
    check_result = check(df)
    assert check_result.check_output.dtype == bool
