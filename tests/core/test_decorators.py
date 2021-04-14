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
    Field,
    Float,
    Int,
    PandasDtype,
    SchemaModel,
    String,
    check_input,
    check_io,
    check_output,
    check_types,
    errors,
)
from pandera.typing import DataFrame, Index, Series


def test_check_function_decorators():
    """
    Tests 5 different methods that are common across the @check_input and
    @check_output decorators.
    """
    in_schema = DataFrameSchema(
        {
            "a": Column(
                Int,
                [
                    Check(lambda x: x >= 1, element_wise=True),
                    Check(lambda s: s.mean() > 0),
                ],
            ),
            "b": Column(
                String,
                Check(lambda x: x in ["x", "y", "z"], element_wise=True),
            ),
            "c": Column(
                DateTime,
                Check(
                    lambda x: pd.Timestamp("2018-01-01") <= x,
                    element_wise=True,
                ),
            ),
            "d": Column(
                Float,
                Check(lambda x: np.isnan(x) or x < 3, element_wise=True),
                nullable=True,
            ),
        },
    )
    out_schema = DataFrameSchema(
        {
            "e": Column(String, Check(lambda s: s == "foo")),
            "f": Column(
                String, Check(lambda x: x in ["a", "b"], element_wise=True)
            ),
        }
    )

    # case 1: simplest path test - df is first argument and function returns
    # single dataframe as output.
    @check_input(in_schema)
    @check_output(out_schema)
    def test_func1(dataframe, x):
        # pylint: disable=W0613
        # disables unused-arguments because handling the second argument is
        # what is being tested.
        return dataframe.assign(e="foo", f=["a", "b", "a"])

    # case 2: input and output validation using positional arguments
    @check_input(in_schema, 1)
    @check_output(out_schema, 0)
    def test_func2(x, dataframe):
        return dataframe.assign(e="foo", f=["a", "b", "a"]), x

    # case 3: dataframe to validate is called as a keyword argument and the
    # output is in a dictionary
    @check_input(in_schema, "in_dataframe")
    @check_output(out_schema, "out_dataframe")
    def test_func3(x, in_dataframe=None):
        return {
            "x": x,
            "out_dataframe": in_dataframe.assign(e="foo", f=["a", "b", "a"]),
        }

    # case 4: dataframe is a positional argument but the obj_getter in the
    # check_input decorator refers to the argument name of the dataframe
    @check_input(in_schema, "dataframe")
    @check_output(out_schema)
    def test_func4(x, dataframe):
        # pylint: disable=W0613
        # disables unused-arguments because handling the second argument is
        # what is being tested.
        return dataframe.assign(e="foo", f=["a", "b", "a"])

    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": ["x", "y", "z"],
            "c": [
                pd.Timestamp("2018-01-01"),
                pd.Timestamp("2018-01-03"),
                pd.Timestamp("2018-01-02"),
            ],
            "d": [np.nan, 1.0, 2.0],
        }
    )

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
        match=r"^error in check_input decorator of function",
    ):
        test_func(pd.DataFrame({"column2": ["a", "b", "c"]}))

    with pytest.raises(
        errors.SchemaError,
        match=r"^error in check_input decorator of function",
    ):
        test_func(df=pd.DataFrame({"column2": ["a", "b", "c"]}))

    with pytest.raises(
        errors.SchemaError,
        match=r"^error in check_output decorator of function",
    ):
        test_func(pd.DataFrame({"column1": [1, 2, 3]}))

    # case 2: check that if the input decorator refers to an index that's not
    # in the function signature, it will fail in a way that's easy to interpret
    @check_input(DataFrameSchema({"column1": Column(Int)}), 1)
    def test_incorrect_check_input_index(df):
        return df

    with pytest.raises(
        IndexError, match=r"^error in check_input decorator of function"
    ):
        test_incorrect_check_input_index(pd.DataFrame({"column1": [1, 2, 3]}))


def test_check_input_method_decorators():
    """Test the check_input and check_output decorator behaviours when the
    dataframe is changed within the function being checked"""
    in_schema = DataFrameSchema({"column1": Column(String)})
    out_schema = DataFrameSchema({"column2": Column(Int)})
    dataframe = pd.DataFrame({"column1": ["a", "b", "c"]})

    def _transform_helper(df):
        return df.assign(column2=[1, 2, 3])

    class TransformerClass:
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
    _assert_expectation(
        transformer.transform_first_arg_with_two_func_args(
            x="foo", df=dataframe
        )
    )

    _assert_expectation(
        transformer.transform_first_arg_with_list_getter(dataframe)
    )
    _assert_expectation(
        transformer.transform_secord_arg_with_list_getter(None, dataframe)
    )
    _assert_expectation(
        transformer.transform_secord_arg_with_dict_getter(None, dataframe)
    )


def test_check_io():
    # pylint: disable=too-many-locals
    """Test that check_io correctly validates/invalidates data."""

    schema = DataFrameSchema({"col": Column(Int, Check.gt(0))})

    @check_io(df1=schema, df2=schema, out=schema)
    def simple_func(df1, df2):
        return df1.assign(col=df1["col"] + df2["col"])

    @check_io(df1=schema, df2=schema)
    def simple_func_no_out(df1, df2):
        return df1.assign(col=df1["col"] + df2["col"])

    @check_io(out=(1, schema))
    def output_with_obj_getter(df):
        return None, df

    @check_io(out=[(0, schema), (1, schema)])
    def multiple_outputs_tuple(df):
        return df, df

    @check_io(
        out=[(0, schema), ("foo", schema), (lambda x: x[2]["bar"], schema)]
    )
    def multiple_outputs_dict(df):
        return {0: df, "foo": df, 2: {"bar": df}}

    @check_io(df=schema, out=schema, head=1)
    def validate_head(df):
        return df

    @check_io(df=schema, out=schema, tail=1)
    def validate_tail(df):
        return df

    @check_io(df=schema, out=schema, sample=1, random_state=100)
    def validate_sample(df):
        return df

    @check_io(df=schema, out=schema, lazy=True)
    def validate_lazy(df):
        return df

    @check_io(df=schema, out=schema, inplace=True)
    def validate_inplace(df):
        return df

    df1 = pd.DataFrame({"col": [1, 1, 1]})
    df2 = pd.DataFrame({"col": [2, 2, 2]})
    invalid_df = pd.DataFrame({"col": [-1, -1, -1]})
    expected = pd.DataFrame({"col": [3, 3, 3]})

    for fn, valid, invalid, out in [
        (simple_func, [df1, df2], [invalid_df, invalid_df], expected),
        (simple_func_no_out, [df1, df2], [invalid_df, invalid_df], expected),
        (output_with_obj_getter, [df1], [invalid_df], (None, df1)),
        (multiple_outputs_tuple, [df1], [invalid_df], (df1, df1)),
        (
            multiple_outputs_dict,
            [df1],
            [invalid_df],
            {0: df1, "foo": df1, 2: {"bar": df1}},
        ),
        (validate_head, [df1], [invalid_df], df1),
        (validate_tail, [df1], [invalid_df], df1),
        (validate_sample, [df1], [invalid_df], df1),
        (validate_lazy, [df1], [invalid_df], df1),
        (validate_inplace, [df1], [invalid_df], df1),
    ]:
        result = fn(*valid)
        if isinstance(result, pd.Series):
            assert (result == out).all()
        if isinstance(result, pd.DataFrame):
            assert (result == out).all(axis=None)
        else:
            assert result == out

        expected_error = (
            errors.SchemaErrors if fn is validate_lazy else errors.SchemaError
        )
        with pytest.raises(expected_error):
            fn(*invalid)

    # invalid out schema types
    for out_schema in [1, 5.0, "foo", {"foo": "bar"}, ["foo"]]:

        @check_io(out=out_schema)
        def invalid_out_schema_type(df):
            return df

        with pytest.raises((TypeError, ValueError)):
            invalid_out_schema_type(df1)


@pytest.mark.parametrize(
    "obj_getter", [1.5, 0.1, ["foo"], {1, 2, 3}, {"foo": "bar"}]
)
def test_check_input_output_unrecognized_obj_getter(obj_getter):
    """
    Test that check_input and check_output raise correct errors on unrecognized
    dataframe object getters
    """
    schema = DataFrameSchema({"column": Column(int)})

    @check_input(schema, obj_getter)
    def test_check_input_fn(df):
        return df

    @check_output(schema, obj_getter)
    def test_check_output_fn(df):
        return df

    for fn in [test_check_input_fn, test_check_output_fn]:
        with pytest.raises(TypeError):
            fn(pd.DataFrame({"column": [1, 2, 3]}))


@pytest.mark.parametrize(
    "out,error,msg",
    [
        (1, TypeError, None),
        (1.5, TypeError, None),
        ("foo", TypeError, None),
        (["foo"], ValueError, "too many values to unpack"),
        (
            (None, "foo"),
            AttributeError,
            "'str' object has no attribute 'validate'",
        ),
        (
            [(None, "foo")],
            AttributeError,
            "'str' object has no attribute 'validate'",
        ),
    ],
)
def test_check_io_unrecognized_obj_getter(out, error, msg):
    """
    Test that check_io raise correct errors on unrecognized decorator arguments
    """

    @check_io(out=out)
    def test_check_io_fn(df):
        return df

    with pytest.raises(error, match=msg):
        test_check_io_fn(pd.DataFrame({"column": [1, 2, 3]}))


# required to be a global: see https://pydantic-docs.helpmanual.io/usage/postponed_annotations/
class OnlyZeroesSchema(SchemaModel):  # pylint:disable=too-few-public-methods
    """Schema with a single column containing zeroes."""

    a: Series[int] = Field(eq=0)


def test_check_types_arguments():
    """Test that check_types forwards key-words arguments to validate."""
    df = pd.DataFrame({"a": [0, 0]})

    @check_types()
    def transform_empty_parenthesis(
        df: DataFrame[OnlyZeroesSchema],
    ) -> DataFrame[OnlyZeroesSchema]:  # pylint: disable=unused-argument
        return df

    transform_empty_parenthesis(df)

    @check_types(head=1)
    def transform_head(
        df: DataFrame[OnlyZeroesSchema],  # pylint: disable=unused-argument
    ) -> DataFrame[OnlyZeroesSchema]:
        return pd.DataFrame({"a": [0, 0]})

    transform_head(df)

    @check_types(tail=1)
    def transform_tail(
        df: DataFrame[OnlyZeroesSchema],  # pylint: disable=unused-argument
    ) -> DataFrame[OnlyZeroesSchema]:
        return pd.DataFrame({"a": [1, 0]})

    transform_tail(df)

    @check_types(lazy=True)
    def transform_lazy(
        df: DataFrame[OnlyZeroesSchema],  # pylint: disable=unused-argument
    ) -> DataFrame[OnlyZeroesSchema]:
        return pd.DataFrame({"a": [1, 1]})

    with pytest.raises(errors.SchemaErrors, match="Usage Tip"):
        transform_lazy(df)


def test_check_types_unchanged():
    """Test the check_types behaviour when the dataframe is unchanged within the
    function being checked."""

    @check_types
    def transform(
        df: DataFrame[OnlyZeroesSchema],
        notused: int,  # pylint: disable=unused-argument
    ) -> DataFrame[OnlyZeroesSchema]:
        return df

    df = pd.DataFrame({"a": [0]})
    pd.testing.assert_frame_equal(transform(df, 2), df)


# required to be globals: see https://pydantic-docs.helpmanual.io/usage/postponed_annotations/
class InSchema(SchemaModel):  # pylint:disable=too-few-public-methods
    """Test schema used as input."""

    a: Series[int]
    idx: Index[str]


class DerivedOutSchema(InSchema):
    """Test schema derived from InSchema."""

    b: Series[int]


class OutSchema(SchemaModel):  # pylint: disable=too-few-public-methods
    """Test schema used as output."""

    b: Series[int]

    class Config:  # pylint: disable=too-few-public-methods
        """Set coerce."""

        coerce = True


def test_check_types_multiple_inputs():
    """Test that check_types behaviour when multiple inputs are annotated."""

    @check_types
    def transform(df_1: DataFrame[InSchema], df_2: DataFrame[InSchema]):
        return pd.concat([df_1, df_2])

    correct = pd.DataFrame({"a": [1]}, index=["1"])
    transform(correct, correct)

    wrong = pd.DataFrame({"b": [1]})
    with pytest.raises(
        errors.SchemaError, match="column 'a' not in dataframe"
    ):
        transform(correct, wrong)


def test_check_types_error_input():
    """Test that check_types raises an error when the input is not correct."""

    @check_types
    def transform(df: DataFrame[InSchema]):
        return df

    df = pd.DataFrame({"b": [1]})
    with pytest.raises(
        errors.SchemaError, match="column 'a' not in dataframe"
    ):
        transform(df)

    try:
        transform(df)
    except errors.SchemaError as exc:
        assert exc.schema == InSchema.to_schema()
        assert exc.data.equals(df)


@pytest.mark.parametrize("out_schema_cls", [DerivedOutSchema, OutSchema])
def test_check_types_error_output(out_schema_cls):
    """Test that check_types raises an error when the output is not correct."""

    df = pd.DataFrame({"a": [1]}, index=["1"])

    @check_types
    def transform(df: DataFrame[InSchema]) -> DataFrame[out_schema_cls]:
        return df

    with pytest.raises(
        errors.SchemaError, match="column 'b' not in dataframe"
    ):
        transform(df)

    try:
        transform(df)
    except errors.SchemaError as exc:
        assert exc.schema == out_schema_cls.to_schema()
        assert exc.data.equals(df)


@pytest.mark.parametrize("out_schema_cls", [DerivedOutSchema, OutSchema])
def test_check_types_optional_out(out_schema_cls):
    """Test the check_types behaviour when the output schema is optional."""

    @check_types
    def optional_out(
        df: DataFrame[InSchema],  # pylint: disable=unused-argument
    ) -> Optional[DataFrame[out_schema_cls]]:
        return None

    df = pd.DataFrame({"a": [1]}, index=["1"])
    assert optional_out(df) is None


def test_check_types_optional_in():
    """Test the check_types behaviour when the input schema is optional."""

    @check_types
    def optional_in(
        df: Optional[DataFrame[InSchema]],  # pylint: disable=unused-argument
    ) -> None:
        return None

    assert optional_in(None) is None


@pytest.mark.parametrize("out_schema_cls", [DerivedOutSchema, OutSchema])
def test_check_types_optional_in_out(out_schema_cls):
    """Test the check_types behaviour when both input and outputs schemas are optional."""

    @check_types
    def transform(
        df: Optional[DataFrame[InSchema]],  # pylint: disable=unused-argument
    ) -> Optional[DataFrame[out_schema_cls]]:
        return None

    assert transform(None) is None


def test_check_types_coerce():
    """Test that check_types return the result of validate."""

    @check_types()
    def transform() -> DataFrame[OutSchema]:
        # OutSchema.b should be coerced to an integer.
        return pd.DataFrame({"b": ["1"]})

    df = transform()
    expected = OutSchema.to_schema().columns["b"].pandas_dtype
    assert PandasDtype(str(df["b"].dtype)) == expected == PandasDtype("int")
