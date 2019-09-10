"""Some unit tests."""

import numpy as np
import pandas as pd
import pytest

from pandera import errors
from pandera import Column, DataFrameSchema, Index, MultiIndex, \
    SeriesSchema, Bool, Category, Check, DateTime, Float, Int, Object, \
    String, Timedelta, check_input, check_output, Hypothesis
from pandera import dtypes
from scipy import stats


def test_column():
    schema = DataFrameSchema({
        "a": Column(Int, Check(lambda x: x > 0, element_wise=True))
    })
    data = pd.DataFrame({"a": [1, 2, 3]})
    assert isinstance(schema.validate(data), pd.DataFrame)


def test_series_schema():
    schema = SeriesSchema(
        Int, Check(lambda x: 0 <= x <= 100, element_wise=True))
    validated_series = schema.validate(pd.Series([0, 30, 50, 100]))
    assert isinstance(validated_series, pd.Series)

    # error cases
    for data in [-1, 101, 50.1, "foo"]:
        with pytest.raises(errors.SchemaError):
            schema.validate(pd.Series([data]))

    for data in [-1, {"a": 1}, -1.0]:
        with pytest.raises(TypeError):
            schema.validate(TypeError)

    non_duplicate_schema = SeriesSchema(
        Int, allow_duplicates=False)
    with pytest.raises(errors.SchemaError):
        non_duplicate_schema.validate(pd.Series([0, 1, 2, 3, 4, 1]))


def test_vectorized_checks():
    schema = SeriesSchema(
        Int, Check(
            lambda s: s.value_counts() == 2, element_wise=False))
    validated_series = schema.validate(pd.Series([1, 1, 2, 2, 3, 3]))
    assert isinstance(validated_series, pd.Series)

    # error case
    with pytest.raises(errors.SchemaError):
        schema.validate(pd.Series([1, 2, 3]))


def test_series_schema_multiple_validators():
    schema = SeriesSchema(
        Int, [
            Check(lambda x: 0 <= x <= 50, element_wise=True),
            Check(lambda s: (s == 21).any())])
    validated_series = schema.validate(pd.Series([1, 5, 21, 50]))
    assert isinstance(validated_series, pd.Series)

    # raise error if any of the validators fails
    with pytest.raises(errors.SchemaError):
        schema.validate(pd.Series([1, 5, 20, 50]))


def test_dataframe_schema():
    schema = DataFrameSchema(
        {
            "a": Column(Int,
                        Check(lambda x: x > 0, element_wise=True)),
            "b": Column(Float,
                        Check(lambda x: 0 <= x <= 10, element_wise=True)),
            "c": Column(String,
                        Check(lambda x: set(x) == {"x", "y", "z"})),
            "d": Column(Bool,
                        Check(lambda x: x.mean() > 0.5)),
            "e": Column(Category,
                        Check(lambda x: set(x) == {"c1", "c2", "c3"})),
            "f": Column(Object,
                        Check(lambda x: x.isin([(1,), (2,), (3,)]))),
            "g": Column(DateTime,
                        Check(lambda x: x >= pd.Timestamp("2015-01-01"),
                              element_wise=True)),
            "i": Column(Timedelta,
                        Check(lambda x: x < pd.Timedelta(10, unit="D"),
                              element_wise=True))
        })
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [1.1, 2.5, 9.9],
        "c": ["z", "y", "x"],
        "d": [True, True, False],
        "e": pd.Series(["c2", "c1", "c3"], dtype="category"),
        "f": [(3,), (2,), (1,)],
        "g": [pd.Timestamp("2015-02-01"),
              pd.Timestamp("2015-02-02"),
              pd.Timestamp("2015-02-03")],
        "i": [pd.Timedelta(1, unit="D"),
              pd.Timedelta(5, unit="D"),
              pd.Timedelta(9, unit="D")]
    })
    assert isinstance(schema.validate(df), pd.DataFrame)

    # error case
    with pytest.raises(errors.SchemaError):
        schema.validate(df.drop("a", axis=1))

    with pytest.raises(errors.SchemaError):
        schema.validate(df.assign(a=[-1, -2, -1]))

    # checks if 'a' is converted to float, while schema says int, will a schema
    # error be thrown
    with pytest.raises(errors.SchemaError):
        schema.validate(df.assign(a=[1.7, 2.3, 3.1]))


def test_dataframe_schema_strict():
    # checks if strict=True whether a schema error is raised because 'a' is not
    # present in the dataframe.
    schema = DataFrameSchema({"a": Column(Int, nullable=True)},
                             strict=True)
    df = pd.DataFrame({"b": [1, 2, 3]})
    with pytest.raises(errors.SchemaError):
        schema.validate(df)


def series_greater_than_zero(s: pd.Series):
    """Return a bool series indicating whether the elements of s are > 0"""
    return s > 0


def series_greater_than_ten(s: pd.Series):
    """Return a bool series indicating whether the elements of s are > 10"""
    return s > 10


class SeriesGreaterCheck:
    """Class creating callable objects to check if series elements exceed a
    lower bound.
    """
    def __init__(self, lower_bound):
        self.lower_bound = lower_bound

    def __call__(self, s: pd.Series):
        """Check if the elements of s are > lower_bound.

        :returns Series with bool elements
        """
        return s > self.lower_bound


@pytest.mark.parametrize("check_function, should_fail", [
    (lambda s: s > 0, False),
    (lambda s: s > 10, True),
    (series_greater_than_zero, False),
    (series_greater_than_ten, True),
    (SeriesGreaterCheck(lower_bound=0), False),
    (SeriesGreaterCheck(lower_bound=10), True)
])
def test_dataframe_schema_check_function_types(check_function, should_fail):
    schema = DataFrameSchema({
            "a": Column(Int,
                        Check(fn=check_function, element_wise=False)),
            "b": Column(Float,
                        Check(fn=check_function, element_wise=False))
    })
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [1.1, 2.5, 9.9]
    })
    if should_fail:
        with pytest.raises(errors.SchemaError):
            schema.validate(df)
    else:
        schema.validate(df)


def test_index_schema():
    schema = DataFrameSchema(
        columns={},
        index=Index(
            Int, [
                Check(lambda x: 1 <= x <= 11, element_wise=True),
                Check(lambda index: index.mean() > 1)]
        ))
    df = pd.DataFrame(index=range(1, 11), dtype="int64")
    assert isinstance(schema.validate(df), pd.DataFrame)

    with pytest.raises(errors.SchemaError):
        schema.validate(pd.DataFrame(index=range(1, 20)))


def test_check_function_decorators():
    in_schema = DataFrameSchema(
        {
            "a": Column(Int, [
                Check(lambda x: x >= 1, element_wise=True),
                Check(lambda s: s.mean() > 0)]),
            "b": Column(String,
                        Check(lambda x: x in ["x", "y", "z"],
                              element_wise=True)),
            "c": Column(DateTime,
                        Check(lambda x: pd.Timestamp("2018-01-01") <= x,
                              element_wise=True)),
            "d": Column(Float,
                        Check(lambda x: np.isnan(x) or x < 3,
                              element_wise=True),
                        nullable=True)
        },
        transformer=lambda df: df.assign(e="foo")
    )
    out_schema = DataFrameSchema(
        {
            "e": Column(String,
                        Check(lambda s: s == "foo")),
            "f": Column(String,
                        Check(lambda x: x in ["a", "b"], element_wise=True))
        })

    # case 1: simplest path test - df is first argument and function returns
    # single dataframe as output.
    @check_input(in_schema)
    @check_output(out_schema)
    def test_func1(dataframe, x):
        return dataframe.assign(f=["a", "b", "a"])

    # case 2: input and output validation using positional arguments
    @check_input(in_schema, 1)
    @check_output(out_schema, 0)
    def test_func2(x, dataframe):
        return dataframe.assign(f=["a", "b", "a"]), x

    # case 3: dataframe to validate is called as a keyword argument and the
    # output is in a dictionary
    @check_input(in_schema, "in_dataframe")
    @check_output(out_schema, "out_dataframe")
    def test_func3(x, in_dataframe=None):
        return {
            "x": x,
            "out_dataframe": in_dataframe.assign(f=["a", "b", "a"]),
        }

    # case 4: dataframe is a positional argument but the obj_getter in the
    # check_input decorator refers to the argument name of the dataframe
    @check_input(in_schema, "dataframe")
    @check_output(out_schema)
    def test_func4(x, dataframe):
        return dataframe.assign(f=["a", "b", "a"])

    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": ["x", "y", "z"],
        "c": [pd.Timestamp("2018-01-01"),
              pd.Timestamp("2018-01-03"),
              pd.Timestamp("2018-01-02")],
        "d": [np.nan, 1.0, 2.0],
    })
    df = test_func1(df, "foo")
    assert isinstance(df, pd.DataFrame)

    df, x = test_func2("foo", df)
    assert x == "foo"
    assert isinstance(df, pd.DataFrame)

    result = test_func3("foo", in_dataframe=df)
    assert result["x"] == "foo"
    assert isinstance(df, pd.DataFrame)

    # case: even if the pandas object to validate is called as a positional
    # argument, the check_input decorator should still be able to handle
    # it.
    result = test_func3("foo", df)
    assert result["x"] == "foo"
    assert isinstance(df, pd.DataFrame)

    df = test_func4("foo", df)
    assert x == "foo"
    assert isinstance(df, pd.DataFrame)


def test_check_function_decorator_errors():
    """Test that the check_input and check_output decorators error properly."""
    # case 1: checks that the input and output decorators error when different
    # types are passed in and out
    @check_input(DataFrameSchema({"column1": Column(Int)}))
    @check_output(DataFrameSchema({"column2": Column(Float)}))
    def test_func(df):
        return df

    with pytest.raises(
            errors.SchemaError,
            match=r"^error in check_input decorator of function"):
        test_func(pd.DataFrame({"column2": ["a", "b", "c"]}))

    with pytest.raises(
            errors.SchemaError,
            match=r"^error in check_output decorator of function"):
        test_func(pd.DataFrame({"column1": [1, 2, 3]}))

    # case 2: check that if the input decorator refers to an index that's not
    # in the function signature, it will fail in a way that's easy to interpret
    @check_input(DataFrameSchema({"column1": Column(Int)}), 1)
    def test_incorrect_check_input_index(df):
        return df

    with pytest.raises(
            errors.SchemaError,
            match=r"^error in check_input decorator of function"
            ):
        test_incorrect_check_input_index(pd.DataFrame({"column1": [1, 2, 3]})
                                         )


def test_check_function_decorator_transform():
    """Test that transformer argument is in effect in check_input decorator."""

    in_schema = DataFrameSchema(
        {"column1": Column(Int)},
        transformer=lambda df: df.assign(column2="foo"))
    out_schema = DataFrameSchema(
        {"column1": Column(Int),
         "column2": Column(String)})

    @check_input(in_schema)
    @check_output(out_schema)
    def func_input_transform1(df):
        return df

    result1 = func_input_transform1(pd.DataFrame({"column1": [1, 2, 3]}))
    assert "column2" in result1

    @check_input(in_schema, 1)
    @check_output(out_schema, 1)
    def func_input_transform2(_, df):
        return _, df

    result2 = func_input_transform2(None, pd.DataFrame({"column1": [1, 2, 3]}))
    assert "column2" in result2[1]


def test_check_input_method_decorators():
    in_schema = DataFrameSchema({"column1": Column(String)})
    out_schema = DataFrameSchema({"column2": Column(Int)})
    dataframe = pd.DataFrame({"column1": ["a", "b", "c"]})

    def _transform_helper(df):
        return df.assign(column2=[1, 2, 3])

    class TransformerClass(object):

        @check_input(in_schema)
        @check_output(out_schema)
        def transform_first_arg(self, df):
            return _transform_helper(df)

        @check_input(in_schema, 0)
        @check_output(out_schema)
        def transform_first_arg_with_list_getter(self, df):
            return _transform_helper(df)

        @check_input(in_schema, 1)
        @check_output(out_schema)
        def transform_secord_arg_with_list_getter(self, x, df):
            return _transform_helper(df)

        @check_input(in_schema, "df")
        @check_output(out_schema)
        def transform_secord_arg_with_dict_getter(self, x, df):
            return _transform_helper(df)

    def _assert_expectation(result_df):
        assert isinstance(result_df, pd.DataFrame)
        assert "column2" in result_df.columns

    transformer = TransformerClass()
    _assert_expectation(transformer.transform_first_arg(dataframe))
    _assert_expectation(
        transformer.transform_first_arg_with_list_getter(dataframe))
    _assert_expectation(
        transformer.transform_secord_arg_with_list_getter(None, dataframe))
    _assert_expectation(
        transformer.transform_secord_arg_with_dict_getter(None, dataframe))


def test_dtypes():
    for dtype in [
            dtypes.Float,
            dtypes.Float16,
            dtypes.Float32,
            dtypes.Float64]:
        schema = DataFrameSchema({"col": Column(dtype, nullable=False)})
        validated_df = schema.validate(
            pd.DataFrame(
                {"col": [-123.1, -7654.321, 1.0, 1.1, 1199.51, 5.1, 4.6]},
                dtype=dtype.value))
        assert isinstance(validated_df, pd.DataFrame)

    for dtype in [
            dtypes.Int,
            dtypes.Int8,
            dtypes.Int16,
            dtypes.Int32,
            dtypes.Int64]:
        schema = DataFrameSchema({"col": Column(dtype, nullable=False)})
        validated_df = schema.validate(
            pd.DataFrame(
                {"col": [-712, -4, -321, 0, 1, 777, 5, 123, 9000]},
                dtype=dtype.value))
        assert isinstance(validated_df, pd.DataFrame)

    for dtype in [
            dtypes.UInt8,
            dtypes.UInt16,
            dtypes.UInt32,
            dtypes.UInt64]:
        schema = DataFrameSchema({"col": Column(dtype, nullable=False)})
        validated_df = schema.validate(
            pd.DataFrame(
                {"col": [1, 777, 5, 123, 9000]}, dtype=dtype.value))
        assert isinstance(validated_df, pd.DataFrame)


def test_nullable_int():
    df = pd.DataFrame({"column1": [5, 1, np.nan]})
    null_schema = DataFrameSchema({
        "column1": Column(Int, Check(lambda x: x > 0), nullable=True)
    })
    assert isinstance(null_schema.validate(df), pd.DataFrame)

    # test case where column is an object
    df = df.astype({"column1": "object"})
    assert isinstance(null_schema.validate(df), pd.DataFrame)


def test_coerce_dtype():
    df = pd.DataFrame({
        "column1": [10.0, 20.0, 30.0],
        "column2": ["2018-01-01", "2018-02-01", "2018-03-01"],
        "column3": [1, 2, 3],
        "column4": [1., 1., np.nan],
    })
    # specify `coerce` at the Column level
    schema1 = DataFrameSchema({
        "column1": Column(Int, Check(lambda x: x > 0), coerce=True),
        "column2": Column(DateTime, coerce=True),
        "column3": Column(String, coerce=True),
    })
    # specify `coerce` at the DataFrameSchema level
    schema2 = DataFrameSchema({
        "column1": Column(Int, Check(lambda x: x > 0)),
        "column2": Column(DateTime),
        "column3": Column(String),
    }, coerce=True)

    for schema in [schema1, schema2]:
        result = schema.validate(df)
        assert result.column1.dtype == Int.value
        assert result.column2.dtype == DateTime.value
        for _, x in result.column3.iteritems():
            assert isinstance(x, str)

        # make sure that correct error is raised when null values are present
        # in a float column that's coerced to an int
        schema = DataFrameSchema({
            "column4": Column(Int, coerce=True)
        })
        with pytest.raises(ValueError):
            schema.validate(df)


def test_required():
    schema = DataFrameSchema({
        "col1": Column(Int, required=False),
        "col2": Column(String)
    })

    df_ok_1 = pd.DataFrame({
        "col2": ['hello', 'world']
    })

    df = schema.validate(df_ok_1)
    assert isinstance(df, pd.DataFrame)
    assert len(df.columns) == 1
    assert set(df.columns) == {"col2"}

    df_ok_2 = pd.DataFrame({
        "col1": [1, 2],
        "col2": ['hello', 'world']
    })

    df = schema.validate(df_ok_2)
    assert isinstance(df, pd.DataFrame)
    assert len(df.columns) == 2
    assert set(df.columns) == {"col1", "col2"}

    df_not_ok = pd.DataFrame({
        "col1": [1, 2]
    })

    with pytest.raises(Exception):
        schema.validate(df_not_ok)


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


def test_hypothesis():
    # Example df for tests:
    df = (
        pd.DataFrame({
            "height_in_feet": [6.5, 7, 6.1, 5.1, 4],
            "sex": ["M", "M", "F", "F", "F"]
        })
    )

    # Initialise the different ways of calling a test:
    schema_pass_ttest_on_alpha_val_1 = DataFrameSchema({
        "height_in_feet": Column(Float, [
            Hypothesis.two_sample_ttest(
                sample1="M",
                sample2="F",
                groupby="sex",
                relationship="greater_than",
                alpha=0.5),
        ]),
        "sex": Column(String)
    })

    schema_pass_ttest_on_alpha_val_2 = DataFrameSchema({
        "height_in_feet": Column(Float, [
            Hypothesis(test=stats.ttest_ind,
                       samples=["M", "F"],
                       groupby="sex",
                       relationship="greater_than",
                       relationship_kwargs={"alpha": 0.5}
                       ),
        ]),
        "sex": Column(String)
    })

    schema_pass_ttest_on_alpha_val_3 = DataFrameSchema({
        "height_in_feet": Column(Float, [
            Hypothesis.two_sample_ttest(
                sample1="M",
                sample2="F",
                groupby="sex",
                relationship="greater_than",
                alpha=0.5),
        ]),
        "sex": Column(String)
    })

    schema_pass_ttest_on_custom_relationship = DataFrameSchema({
        "height_in_feet": Column(Float, [
            Hypothesis(
                test=stats.ttest_ind,
                samples=["M", "F"],
                groupby="sex",
                relationship=lambda stat, pvalue, alpha=0.01: (
                    stat > 0 and pvalue / 2 < alpha
                ),
                relationship_kwargs={"alpha": 0.5}
            )
        ]),
        "sex": Column(String),
    })

    # Check the 3 happy paths are successful:
    schema_pass_ttest_on_alpha_val_1.validate(df)
    schema_pass_ttest_on_alpha_val_2.validate(df)
    schema_pass_ttest_on_alpha_val_3.validate(df)
    schema_pass_ttest_on_custom_relationship.validate(df)

    schema_fail_ttest_on_alpha_val_1 = DataFrameSchema({
        "height_in_feet": Column(Float, [
            Hypothesis.two_sample_ttest(
                sample1="M",
                sample2="F",
                groupby="sex",
                relationship="greater_than",
                alpha=0.05),
        ]),
        "sex": Column(String)
    })

    schema_fail_ttest_on_alpha_val_2 = DataFrameSchema({
        "height_in_feet": Column(Float, [
            Hypothesis(test=stats.ttest_ind,
                       samples=["M", "F"],
                       groupby="sex",
                       relationship="greater_than",
                       relationship_kwargs={"alpha": 0.05}),
        ]),
        "sex": Column(String)
    })

    schema_fail_ttest_on_alpha_val_3 = DataFrameSchema({
        "height_in_feet": Column(Float, [
            Hypothesis.two_sample_ttest(
                sample1="M",
                sample2="F",
                groupby="sex",
                relationship="greater_than",
                alpha=0.05),
        ]),
        "sex": Column(String)
    })

    with pytest.raises(errors.SchemaError):
        schema_fail_ttest_on_alpha_val_1.validate(df)
    with pytest.raises(errors.SchemaError):
        schema_fail_ttest_on_alpha_val_2.validate(df)
    with pytest.raises(errors.SchemaError):
        schema_fail_ttest_on_alpha_val_3.validate(df)


def test_two_sample_ttest_hypothesis_relationships():
    """Check allowable relationships in two-sample ttest."""
    for relationship in Hypothesis.RELATIONSHIPS:
        schema = DataFrameSchema({
            "height_in_feet": Column(Float, [
                Hypothesis.two_sample_ttest(
                    sample1="M",
                    sample2="F",
                    groupby="sex",
                    relationship=relationship,
                    alpha=0.5),
            ]),
            "sex": Column(String)
        })
        assert isinstance(schema, DataFrameSchema)

    for relationship in ["foo", "bar", 1, 2, 3, None]:
        with pytest.raises(errors.SchemaError):
            DataFrameSchema({
                "height_in_feet": Column(Float, [
                    Hypothesis.two_sample_ttest(
                        sample1="M",
                        sample2="F",
                        groupby="sex",
                        relationship=relationship,
                        alpha=0.5),
                ]),
                "sex": Column(String)
            })


def test_multi_index_columns():
    schema = DataFrameSchema({
        ("zero", "foo"): Column(Float, Check(lambda s: (s > 0) & (s < 1))),
        ("zero", "bar"): Column(
            String, Check(lambda s: s.isin(["a", "b", "c", "d"]))),
        ("one", "foo"): Column(Int, Check(lambda s: (s > 0) & (s < 10))),
        ("one", "bar"): Column(
            DateTime, Check(lambda s: s == pd.datetime(2019, 1, 1)))
    })
    validated_df = schema.validate(
        pd.DataFrame({
            ("zero", "foo"): [0.1, 0.2, 0.7, 0.3],
            ("zero", "bar"): ["a", "b", "c", "d"],
            ("one", "foo"): [1, 6, 4, 7],
            ("one", "bar"): pd.to_datetime(["2019/01/01"] * 4)
        })
    )
    assert isinstance(validated_df, pd.DataFrame)


def test_multi_index_index():
    schema = DataFrameSchema(
        columns={
            "column1": Column(Float, Check(lambda s: s > 0)),
            "column2": Column(Float, Check(lambda s: s > 0)),
        },
        index=MultiIndex(
            indexes=[
                Index(Int,
                      Check(lambda s: (s < 5) & (s >= 0)),
                      name="index0"),
                Index(String,
                      Check(lambda s: s.isin(["foo", "bar"])),
                      name="index1"),
            ]
        )
    )

    df = pd.DataFrame(
        data={
            "column1": [0.1, 0.5, 123.1, 10.6, 22.31],
            "column2": [0.1, 0.5, 123.1, 10.6, 22.31],
        },
        index=pd.MultiIndex(
            levels=[[0, 1, 2, 3, 4], ["foo", "bar"]],
            labels=[[0, 1, 2, 3, 4], [0, 1, 0, 1, 0]],
            names=["index0", "index1"],
        )
    )

    validated_df = schema.validate(df)
    assert isinstance(validated_df, pd.DataFrame)

    # failure case
    df_fail = df.copy()
    df_fail.index = pd.MultiIndex(
        levels=[[0, 1, 2, 3, 4], ["foo", "bar"]],
        labels=[[-1, 1, 2, 3, 4], [0, 1, 0, 1, 0]],
        names=["index0", "index1"],
    )
    with pytest.raises(errors.SchemaError):
        schema.validate(df_fail)


def test_head_dataframe_schema():
    """
    Test that schema can validate head of dataframe, returns entire dataframe.
    """

    df = pd.DataFrame({
        "col1": [i for i in range(100)] + [i for i in range(-1, -1001, -1)]
    })

    schema = DataFrameSchema(
        columns={"col1": Column(Int, Check(lambda s: s >= 0))})

    # Validating with head of 100 should pass
    assert schema.validate(df, head=100).equals(df)
    with pytest.raises(errors.SchemaError):
        schema.validate(df)


def test_tail_dataframe_schema():
    df = pd.DataFrame({
        "col1": [i for i in range(100)] + [i for i in range(-1, -1001, -1)]
    })

    schema = DataFrameSchema(
        columns={"col1": Column(Int, Check(lambda s: s < 0))})

    # Validating with tail of 1000 should pass
    assert schema.validate(df, tail=1000).equals(df)
    with pytest.raises(errors.SchemaError):
        schema.validate(df)


def test_sample_dataframe_schema():
    df = pd.DataFrame({
        "col1": range(1, 1001)
    })

    # assert all values -1
    schema = DataFrameSchema(
        columns={"col1": Column(Int, Check(lambda s: s == -1))})

    for seed in [11, 123456, 9000, 654]:
        sample_index = df.sample(100, random_state=seed).index
        df.loc[sample_index] = -1
        assert schema.validate(df, sample=100, random_state=seed).equals(df)


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
            Check(lambda g: g["foo"]["col1"].item() == 1, groupby="col3"),
            Check(lambda g: g["foo"]["col2"].item() == 2.0, groupby="col3"),
            Check(lambda g: g["foo"]["col3"].item() == "foo", groupby="col3"),
            Check(lambda g: g[("foo", "foo")]["col1"].item() == 1,
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


def test_dataframe_hypothesis_checks():

    df = pd.DataFrame({
        "col1": range(100, 201),
        "col2": range(0, 101),
    })

    hypothesis_check_schema = DataFrameSchema(
        columns={
            "col1": Column(Int),
            "col2": Column(Int),
        },
        checks=[
            # two-sample test
            Hypothesis(
                test=stats.ttest_ind,
                samples=["col1", "col2"],
                relationship=lambda stat, pvalue, alpha=0.01: (
                    stat > 0 and pvalue / 2 < alpha
                ),
                relationship_kwargs={"alpha": 0.5},
            ),
            # one-sample test
            Hypothesis(
                test=stats.ttest_1samp,
                samples=["col1"],
                relationship=lambda stat, pvalue, alpha=0.01: (
                    stat > 0 and pvalue / 2 < alpha
                ),
                test_kwargs={"popmean": 50},
                relationship_kwargs={"alpha": 0.01},
            ),
        ]
    )

    hypothesis_check_schema.validate(df)

    # raise error when using groupby
    hypothesis_check_schema_groupby = DataFrameSchema(
        columns={
            "col1": Column(Int),
            "col2": Column(Int),
        },
        checks=[
            # two-sample test
            Hypothesis(
                test=stats.ttest_ind,
                samples=["col1", "col2"],
                groupby="col3",
                relationship=lambda stat, pvalue, alpha=0.01: (
                    stat > 0 and pvalue / 2 < alpha
                ),
                relationship_kwargs={"alpha": 0.5},
            ),
        ]
    )
    with pytest.raises(errors.SchemaDefinitionError):
        hypothesis_check_schema_groupby.validate(df)
