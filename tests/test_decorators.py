import numpy as np
import pandas as pd
import pytest

from pandera import errors
from pandera import (
    Column, DataFrameSchema, Check, DateTime, Float, Int, String, check_input,
    check_output)


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

