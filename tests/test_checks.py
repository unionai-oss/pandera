import pandas as pd
import pytest

from pandera import errors
from pandera import (
    Column, DataFrameSchema, Index, SeriesSchema, Bool,
    Check, Float, Int, String)


def test_vectorized_checks():
    schema = SeriesSchema(
        Int, Check(
            lambda s: s.value_counts() == 2, element_wise=False))
    validated_series = schema.validate(pd.Series([1, 1, 2, 2, 3, 3]))
    assert isinstance(validated_series, pd.Series)

    # error case
    with pytest.raises(errors.SchemaError):
        schema.validate(pd.Series([1, 2, 3]))


def test_check_groupby():
    schema = DataFrameSchema({
        "col1": Column(Int, [
            Check(lambda s: s["foo"] > 10, groupby="col2"),
            Check(lambda s: s["bar"] < 10, groupby=["col2"]),
            Check(lambda s: s["foo"] > 10,
                  groupby=lambda df: df.groupby("col2")),
            Check(lambda s: s["bar"] < 10,
                  groupby=lambda df: df.groupby("col2"))
        ]),
        "col2": Column(String, Check(lambda s: s.isin(["foo", "bar"]))),
    })

    df_pass = pd.DataFrame({
        "col1": [7, 8, 9, 11, 12, 13],
        "col2": ["bar", "bar", "bar", "foo", "foo", "foo"],
    })

    df = schema.validate(df_pass)
    assert isinstance(df, pd.DataFrame)
    assert len(df.columns) == 2
    assert set(df.columns) == {"col1", "col2"}

    # raise errors.SchemaError when Check fails
    df_fail_on_bar = pd.DataFrame({
        "col1": [7, 8, 20, 11, 12, 13],
        "col2": ["bar", "bar", "bar", "foo", "foo", "foo"],
    })
    df_fail_on_foo = pd.DataFrame({
        "col1": [7, 8, 9, 11, 1, 13],
        "col2": ["bar", "bar", "bar", "foo", "foo", "foo"],
    })
    # raise errors.SchemaError when groupby column doesn't exist
    df_fail_no_column = pd.DataFrame({
        "col1": [7, 8, 20, 11, 12, 13],
    })

    for df in [df_fail_on_bar, df_fail_on_foo, df_fail_no_column]:
        with pytest.raises(errors.SchemaError):
            schema.validate(df)


def test_check_groupby_multiple_columns():
    schema = DataFrameSchema({
        "col1": Column(Int, [
            Check(lambda s: s[("bar", True)].sum() == 16,  # 7 + 9
                  groupby=["col2", "col3"]),
        ]),
        "col2": Column(String, Check(lambda s: s.isin(["foo", "bar"]))),
        "col3": Column(Bool),
    })

    df_pass = pd.DataFrame({
        "col1": [7, 8, 9, 11, 12, 13],
        "col2": ["bar", "bar", "bar", "foo", "foo", "foo"],
        "col3": [True, False, True, False, True, False],
    })

    df = schema.validate(df_pass)
    assert isinstance(df, pd.DataFrame)
    assert len(df.columns) == 3
    assert set(df.columns) == {"col1", "col2", "col3"}


def test_check_groups():
    schema = DataFrameSchema({
        "col1": Column(Int, [
            Check(lambda s: s["foo"] > 10, groupby="col2", groups=["foo"]),
            Check(lambda s: s["foo"] > 10, groupby="col2", groups="foo"),
        ]),
        "col2": Column(String, Check(lambda s: s.isin(["foo", "bar"]))),
    })

    df = pd.DataFrame({
        "col1": [7, 8, 9, 11, 12, 13],
        "col2": ["bar", "bar", "bar", "foo", "foo", "foo"],
    })

    validated_df = schema.validate(df)
    assert isinstance(validated_df, pd.DataFrame)
    assert len(validated_df.columns) == 2
    assert set(validated_df.columns) == {"col1", "col2"}

    # raise KeyError when groups does not include a particular group name
    schema_fail_key_error = DataFrameSchema({
        "col1": Column(Int, [
            Check(lambda s: s["bar"] > 10, groupby="col2", groups="foo"),
        ]),
        "col2": Column(String, Check(lambda s: s.isin(["foo", "bar"]))),
    })
    with pytest.raises(KeyError, match="^'bar'"):
        schema_fail_key_error.validate(df)

    # raise KeyError when the group does not exist in the groupby column when
    # referenced in the Check function
    schema_fail_nonexistent_key_in_fn = DataFrameSchema({
        "col1": Column(Int, [
            Check(lambda s: s["baz"] > 10, groupby="col2", groups=["foo"]),
        ]),
        "col2": Column(String, Check(lambda s: s.isin(["foo", "bar"]))),
    })
    with pytest.raises(KeyError, match="^'baz'"):
        schema_fail_nonexistent_key_in_fn.validate(df)

    # raise KeyError when the group does not exist in the groups argument.
    schema_fail_nonexistent_key_in_groups = DataFrameSchema({
        "col1": Column(Int, [
            Check(lambda s: s["foo"] > 10, groupby="col2", groups=["baz"]),
        ]),
        "col2": Column(String, Check(lambda s: s.isin(["foo", "bar"]))),
    })
    with pytest.raises(KeyError):
        schema_fail_nonexistent_key_in_groups.validate(df)


def test_groupby_init_exceptions():
    def init_schema_element_wise():
        DataFrameSchema({
            "col1": Column(Int, [
                Check(lambda s: s["foo"] > 10,
                      element_wise=True,
                      groupby=["col2"]),
            ]),
            "col2": Column(String, Check(lambda s: s.isin(["foo", "bar"]))),
        })

    # can't use groupby in Checks where element_wise == True
    with pytest.raises(
            errors.SchemaInitError,
            match=r"^Cannot use groupby when element_wise=True."):
        init_schema_element_wise()

    # raise errors.SchemaInitError even when the schema doesn't specify column
    # key for groupby column
    def init_schema_no_groupby_column():
        DataFrameSchema({
            "col1": Column(Int, [
                Check(lambda s: s["foo"] > 10, groupby=["col2"]),
            ]),
        })

    with pytest.raises(errors.SchemaInitError):
        init_schema_no_groupby_column()

    # can't use groupby argument in SeriesSchema or Index objects
    for SchemaClass in [SeriesSchema, Index]:
        with pytest.raises(
                errors.SchemaInitError,
                match="^Cannot use groupby checks with"):
            SchemaClass(Int, Check(lambda s: s["bar"] == 1, groupby="foo"))


def test_dataframe_checks():
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
        ]
    )
    df = pd.DataFrame({
        "col1": [1, 2, 3],
        "col2": [2.0, 3.0, 4.0],
        "col3": ["foo", "bar", "baz"],
        "col4": ["foo", "bar", "baz"],
    })

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
            Check(lambda g: g[("foo", "foo")]["col1"].iat[0] == 1,
                  groupby=["col3", "col4"]),
        ]
    )
    assert isinstance(groupby_check_schema.validate(df), pd.DataFrame)

    # test element-wise checks
    element_wise_check_schema = DataFrameSchema(
        columns={
            "col1": Column(Int),
            "col2": Column(Float),
        },
        checks=Check(lambda row: row["col1"] < row["col2"], element_wise=True)
    )
    assert isinstance(element_wise_check_schema.validate(df), pd.DataFrame)
