"""Testing the Decorators that check a functions input or output."""

from typing import Optional

import numpy as np
import pandas as pd
import pytest

from pandera import (
    Check,
    Column,
    DataFrameSchema,
    DateTime,
    Float,
    Int,
    SchemaModel,
    String,
    check_input,
    check_output,
    check_types,
    errors,
)
from pandera.typing import DataFrame, Index, Series


def test_check_function_decorators():
    """Tests 5 different methods that are common across the @check_input and
    @check_output decorators.
    """
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
        # pylint: disable=W0613
        # disables unused-arguments because handling the second argument is
        # what is being tested.
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
        # pylint: disable=W0613
        # disables unused-arguments because handling the second argument is
        # what is being tested.
        return dataframe.assign(f=["a", "b", "a"])

    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": ["x", "y", "z"],
        "c": [pd.Timestamp("2018-01-01"),
              pd.Timestamp("2018-01-03"),
              pd.Timestamp("2018-01-02")],
        "d": [np.nan, 1.0, 2.0],
    })

    # call function with a dataframe passed as a positional argument
    df = test_func1(df, "foo")
    assert isinstance(df, pd.DataFrame)

    # call function with a dataframe passed as a first keyword argument
    df = test_func1(dataframe=df, x="foo")
    assert isinstance(df, pd.DataFrame)

    # call function with a dataframe passed as a second keyword argument
    df = test_func1(x="foo", dataframe=df)
    assert isinstance(df, pd.DataFrame)

    df, x = test_func2("foo", df)
    assert x == "foo"
    assert isinstance(df, pd.DataFrame)

    result = test_func3("foo", in_dataframe=df)
    assert result["x"] == "foo"
    assert isinstance(df, pd.DataFrame)

    # case 5: even if the pandas object to validate is called as a positional
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
            match=r"^error in check_input decorator of function"):
        test_func(df=pd.DataFrame({"column2": ["a", "b", "c"]}))

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
            IndexError,
            match=r"^error in check_input decorator of function"):
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
    """Test the check_input and check_output decorator behaviours when the
    dataframe is changed within the function being checked"""
    in_schema = DataFrameSchema({"column1": Column(String)})
    out_schema = DataFrameSchema({"column2": Column(Int)})
    dataframe = pd.DataFrame({"column1": ["a", "b", "c"]})

    def _transform_helper(df):
        return df.assign(column2=[1, 2, 3])

    class TransformerClass():
        """Contains functions with different signatures representing the way
        that the decorators can be called."""
        # pylint: disable=E0012,C0111,C0116,W0613,R0201
        # disables missing-function-docstring as this is a factory method
        # disables unused-arguments because handling the second argument is
        # what is being tested and this is intentional.
        # disables no-self-use because having TransformerClass with functions
        # is cleaner.

        @check_input(in_schema)
        @check_output(out_schema)
        def transform_first_arg(self, df):
            return _transform_helper(df)

        @check_input(in_schema)
        @check_output(out_schema)
        def transform_first_arg_with_two_func_args(self, df, x):
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

    # call method with a dataframe passed as a positional argument
    _assert_expectation(transformer.transform_first_arg(dataframe))

    # call method with a dataframe passed as a first keyword argument
    _assert_expectation(transformer.transform_first_arg(df=dataframe))

    # call method with a dataframe passed as a second keyword argument
    _assert_expectation(transformer.transform_first_arg_with_two_func_args(x="foo", df=dataframe))

    _assert_expectation(
        transformer.transform_first_arg_with_list_getter(dataframe))
    _assert_expectation(
        transformer.transform_secord_arg_with_list_getter(None, dataframe))
    _assert_expectation(
        transformer.transform_secord_arg_with_dict_getter(None, dataframe))

# required to be globals: see https://pydantic-docs.helpmanual.io/usage/postponed_annotations/
class InSchema(SchemaModel): # pylint:disable=R0903:
    """Test schema used as input."""
    a: Series[int]
    idx: Index[str]


class OutSchema(SchemaModel): # pylint:disable=R0903:
    """Test schema used as output."""
    b: Series[int]


def test_check_types_unchanged():
    """Test the check_types behaviour when the dataframe is unchanged within the
    function being checked."""

    @check_types
    def transform(df: DataFrame[InSchema], notused: int) -> DataFrame[InSchema]: # pylint: disable=W0613
        return df

    df = pd.DataFrame({"a": [1]}, index=["1"])
    pd.testing.assert_frame_equal(transform(df, 2), df)


def test_check_types_errors():
    """Test that check_types behaviour raises an error if the input or ouput schemas
    are not respected."""

    df = pd.DataFrame({"a": [1]}, index=["1"])

    @check_types
    def transform_index(df: DataFrame[InSchema]) -> DataFrame[InSchema]:
        return df.reset_index(drop=True)

    with pytest.raises(errors.SchemaError):
        transform_index(df)

    @check_types
    def to_b(df: DataFrame[InSchema]) -> DataFrame[OutSchema]:
        return df

    with pytest.raises(errors.SchemaError, match="column 'b' not in dataframe"):
        to_b(df)

    @check_types
    def to_str(df: DataFrame[InSchema]) -> DataFrame[InSchema]:
        df["a"] = "1"
        return df

    err_msg = "expected series 'a' to have type int64"
    with pytest.raises(errors.SchemaError, match=err_msg):
        to_str(df)


def test_check_types_optional_out():
    """Test the check_types behaviour when the output schema is optional."""

    @check_types
    def optional_out(df: DataFrame[InSchema]) -> Optional[DataFrame[OutSchema]]: # pylint: disable=W0613
        return None

    df = pd.DataFrame({"a": [1]}, index=["1"])
    assert optional_out(df) is None


def test_check_types_optional_in():
    """Test the check_types behaviour when the input schema is optional."""

    @check_types
    def optional_in(df: Optional[DataFrame[InSchema]]) -> None: # pylint: disable=W0613
        return None

    assert optional_in(None) is None


def test_check_types_optional_in_out():
    """Test the check_types behaviour when both input and outputs schemas are optional."""

    @check_types
    def transform(df: Optional[DataFrame[InSchema]]) -> Optional[DataFrame[OutSchema]]: # pylint: disable=W0613
        return None

    assert transform(None) is None
