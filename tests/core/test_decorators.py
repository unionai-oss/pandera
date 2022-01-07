"""Testing the Decorators that check a functions input or output."""
import typing
from asyncio import AbstractEventLoop

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
    SchemaModel,
    String,
    check_input,
    check_io,
    check_output,
    check_types,
    errors,
)
from pandera.engines.pandas_engine import Engine
from pandera.typing import DataFrame, Index, Series

try:
    from typing import Literal  # type: ignore
except ImportError:
    # Remove this after dropping python 3.6
    from typing_extensions import Literal  # type: ignore


def test_check_function_decorators() -> None:
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


def test_check_function_decorator_errors() -> None:
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


def test_check_input_method_decorators() -> None:
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


def test_check_io() -> None:
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
        result = fn(*valid)  # type: ignore[operator]
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
            fn(*invalid)  # type: ignore[operator]

    # invalid out schema types
    for out_schema in [1, 5.0, "foo", {"foo": "bar"}, ["foo"]]:

        # mypy finds correctly the wrong usage
        # pylint: disable=cell-var-from-loop
        @check_io(out=out_schema)  # type: ignore[arg-type]
        def invalid_out_schema_type(df):
            return df

        with pytest.raises((TypeError, ValueError)):
            invalid_out_schema_type(df1)


@pytest.mark.parametrize(
    "obj_getter", [1.5, 0.1, ["foo"], {1, 2, 3}, {"foo": "bar"}]
)
def test_check_input_output_unrecognized_obj_getter(obj_getter) -> None:
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
def test_check_io_unrecognized_obj_getter(out, error, msg) -> None:
    """
    Test that check_io raise correct errors on unrecognized decorator arguments
    """

    @check_io(out=out)
    def test_check_io_fn(df):
        return df

    with pytest.raises(error, match=msg):
        test_check_io_fn(pd.DataFrame({"column": [1, 2, 3]}))


# required to be a global: see
# https://pydantic-docs.helpmanual.io/usage/postponed_annotations/
class OnlyZeroesSchema(SchemaModel):  # pylint:disable=too-few-public-methods
    """Schema with a single column containing zeroes."""

    a: Series[int] = Field(eq=0)


def test_check_types_arguments() -> None:
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


def test_check_types_unchanged() -> None:
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


# required to be globals:
# see https://pydantic-docs.helpmanual.io/usage/postponed_annotations/
class InSchema(SchemaModel):  # pylint:disable=too-few-public-methods
    """Test schema used as input."""

    a: Series[int]
    idx: Index[str]

    class Config:  # pylint: disable=too-few-public-methods
        """Set coerce."""

        coerce = True


class DerivedOutSchema(InSchema):
    """Test schema derived from InSchema."""

    b: Series[int]


class OutSchema(SchemaModel):  # pylint: disable=too-few-public-methods
    """Test schema used as output."""

    b: Series[int]

    class Config:  # pylint: disable=too-few-public-methods
        """Set coerce."""

        coerce = True


def test_check_types_multiple_inputs() -> None:
    """Test that check_types behaviour when multiple inputs are annotated."""

    @check_types
    def transform(df_1: DataFrame[InSchema], df_2: DataFrame[InSchema]):
        return pd.concat([df_1, df_2])

    correct = pd.DataFrame({"a": [1]}, index=["1"])
    transform(correct, df_2=correct)

    wrong = pd.DataFrame({"b": [1]})
    with pytest.raises(
        errors.SchemaError, match="column 'a' not in dataframe"
    ):
        transform(correct, wrong)


def test_check_types_error_input() -> None:
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


def test_check_types_error_output() -> None:
    """Test that check_types raises an error when the output is not correct."""

    df = pd.DataFrame({"a": [1]}, index=["1"])

    @check_types
    def transform_derived(
        df: DataFrame[InSchema],
    ) -> DataFrame[DerivedOutSchema]:
        return df

    with pytest.raises(
        errors.SchemaError, match="column 'b' not in dataframe"
    ):
        transform_derived(df)

    try:
        transform_derived(df)
    except errors.SchemaError as exc:
        assert exc.schema == DerivedOutSchema.to_schema()
        assert exc.data.equals(df)

    df = pd.DataFrame({"a": [1]}, index=["1"])

    @check_types
    def transform(df: DataFrame[InSchema]) -> DataFrame[OutSchema]:
        return df

    with pytest.raises(
        errors.SchemaError, match="column 'b' not in dataframe"
    ):
        transform(df)

    try:
        transform(df)
    except errors.SchemaError as exc:
        assert exc.schema == OutSchema.to_schema()
        assert exc.data.equals(df)


def test_check_types_optional_out() -> None:
    """Test the check_types behaviour when the output schema is Optional."""

    @check_types
    def optional_derived_out(
        df: DataFrame[InSchema],  # pylint: disable=unused-argument
    ) -> typing.Optional[DataFrame[DerivedOutSchema]]:
        return None

    df = pd.DataFrame({"a": [1]}, index=["1"])
    assert optional_derived_out(df) is None

    @check_types
    def optional_out(
        df: DataFrame[InSchema],  # pylint: disable=unused-argument
    ) -> typing.Optional[DataFrame[OutSchema]]:
        return None

    df = pd.DataFrame({"a": [1]}, index=["1"])
    assert optional_out(df) is None


def test_check_types_optional_in() -> None:
    """Test the check_types behaviour when the input schema is Optional."""

    @check_types
    def optional_in(
        # pylint: disable=unused-argument
        df: typing.Optional[DataFrame[InSchema]],
    ) -> None:
        return None

    assert optional_in(None) is None


def test_check_types_optional_in_out() -> None:
    """
    Test the check_types behaviour when both input and outputs schemas are
    Optional.
    """

    @check_types
    def transform_derived(
        # pylint: disable=unused-argument
        df: typing.Optional[DataFrame[InSchema]],
    ) -> typing.Optional[DataFrame[DerivedOutSchema]]:
        return None

    assert transform_derived(None) is None

    @check_types
    def transform(
        # pylint: disable=unused-argument
        df: typing.Optional[DataFrame[InSchema]],
    ) -> typing.Optional[DataFrame[OutSchema]]:
        return None

    assert transform(None) is None


def test_check_types_coerce() -> None:
    """Test that check_types return the result of validate."""

    @check_types()
    def transform_in(df: DataFrame[InSchema]):
        return df

    df = transform_in(pd.DataFrame({"a": ["1"]}, index=["1"]))
    expected = InSchema.to_schema().columns["a"].dtype
    assert Engine.dtype(df["a"].dtype) == expected

    @check_types()
    def transform_out() -> DataFrame[OutSchema]:
        # OutSchema.b should be coerced to an integer.
        return pd.DataFrame({"b": ["1"]})

    out_df = transform_out()
    expected = OutSchema.to_schema().columns["b"].dtype
    assert Engine.dtype(out_df["b"].dtype) == expected


@pytest.mark.parametrize(
    "arg_examples",
    [
        [1, 5, 10, 123],
        list("abcdefg"),
        [1.0, 1.1, 1.3, 10.2],
        [None],
    ],
)
def test_check_types_with_literal_type(arg_examples):
    """Test that using typing module types works with check_types"""

    for example in arg_examples:
        arg_type = Literal[example]

        @check_types
        def transform_with_literal(
            df: DataFrame[InSchema],
            # pylint: disable=unused-argument,cell-var-from-loop
            arg: arg_type,
        ) -> DataFrame[OutSchema]:
            return df.assign(b=100)

        df = pd.DataFrame({"a": [1]})
        invalid_df = pd.DataFrame()

        transform_with_literal(df, example)
        with pytest.raises(errors.SchemaError):
            transform_with_literal(invalid_df, example)


def test_check_types_method_args() -> None:
    """Test that @check_types works with positional and keyword args in methods,
    classmethods and staticmethods.
    """
    # pylint: disable=unused-argument,missing-class-docstring,too-few-public-methods,missing-function-docstring

    class SchemaIn1(SchemaModel):
        col1: Series[int]

        class Config:
            strict = True

    class SchemaIn2(SchemaModel):
        col2: Series[int]

        class Config:
            strict = True

    class SchemaOut(SchemaModel):
        col3: Series[int]

        class Config:
            strict = True

    in1: DataFrame[SchemaIn1] = DataFrame({SchemaIn1.col1: [1]})
    in2: DataFrame[SchemaIn2] = DataFrame({SchemaIn2.col2: [2]})
    out: DataFrame[SchemaOut] = DataFrame({SchemaOut.col3: [3]})

    class SomeClass:
        @check_types
        def regular_method(  # pylint: disable=no-self-use
            self,
            df1: DataFrame[SchemaIn1],
            df2: DataFrame[SchemaIn2],
        ) -> DataFrame[SchemaOut]:
            return out

        @classmethod
        @check_types
        def class_method(
            cls, df1: DataFrame[SchemaIn1], df2: DataFrame[SchemaIn2]
        ) -> DataFrame[SchemaOut]:
            return out

        @staticmethod
        @check_types
        def static_method(
            df1: DataFrame[SchemaIn1], df2: DataFrame[SchemaIn2]
        ) -> DataFrame[SchemaOut]:
            return out

    instance = SomeClass()

    pd.testing.assert_frame_equal(
        out, instance.regular_method(in1, in2)
    )  # Used to fail
    pd.testing.assert_frame_equal(out, instance.regular_method(in1, df2=in2))
    pd.testing.assert_frame_equal(
        out, instance.regular_method(df1=in1, df2=in2)
    )

    with pytest.raises(errors.SchemaError):
        instance.regular_method(in2, in1)  # Used to fail
    with pytest.raises(errors.SchemaError):
        instance.regular_method(in2, df2=in1)
    with pytest.raises(errors.SchemaError):
        instance.regular_method(df1=in2, df2=in1)

    pd.testing.assert_frame_equal(out, SomeClass.class_method(in1, in2))
    pd.testing.assert_frame_equal(out, SomeClass.class_method(in1, df2=in2))
    pd.testing.assert_frame_equal(
        out, SomeClass.class_method(df1=in1, df2=in2)
    )

    with pytest.raises(errors.SchemaError):
        instance.class_method(in2, in1)
    with pytest.raises(errors.SchemaError):
        instance.class_method(in2, df2=in1)
    with pytest.raises(errors.SchemaError):
        instance.class_method(df1=in2, df2=in1)

    pd.testing.assert_frame_equal(out, instance.static_method(in1, in2))
    pd.testing.assert_frame_equal(out, SomeClass.static_method(in1, in2))
    pd.testing.assert_frame_equal(out, instance.static_method(in1, df2=in2))
    pd.testing.assert_frame_equal(out, SomeClass.static_method(in1, df2=in2))
    pd.testing.assert_frame_equal(
        out, instance.static_method(df1=in1, df2=in2)
    )

    with pytest.raises(errors.SchemaError):
        instance.static_method(in2, in1)
    with pytest.raises(errors.SchemaError):
        instance.static_method(in2, df2=in1)
    with pytest.raises(errors.SchemaError):
        instance.static_method(df1=in2, df2=in1)


def test_coroutines(event_loop: AbstractEventLoop) -> None:
    # pylint: disable=missing-class-docstring,too-few-public-methods,missing-function-docstring
    class Schema(SchemaModel):
        col1: Series[int]

        class Config:
            strict = True

    @check_types
    @check_output(Schema.to_schema())
    @check_input(Schema.to_schema())
    @check_io(df1=Schema.to_schema(), out=Schema.to_schema())
    async def coroutine(df1: DataFrame[Schema]) -> DataFrame[Schema]:
        return df1

    class Meta(type):
        @check_types
        @check_output(Schema.to_schema())
        @check_input(Schema.to_schema(), "df1")
        @check_io(df1=Schema.to_schema(), out=Schema.to_schema())
        async def regular_meta_coroutine(  # pylint: disable=no-self-use
            cls,
            df1: DataFrame[Schema],
        ) -> DataFrame[Schema]:
            return df1

        @classmethod
        @check_types
        @check_output(Schema.to_schema())
        @check_input(Schema.to_schema(), "df1")
        @check_io(df1=Schema.to_schema(), out=Schema.to_schema())
        async def class_meta_coroutine(  # pylint: disable=bad-mcs-classmethod-argument
            mcs, df1: DataFrame[Schema]
        ) -> DataFrame[Schema]:
            return df1

        @staticmethod
        @check_types
        @check_output(Schema.to_schema())
        @check_input(Schema.to_schema())
        @check_io(df1=Schema.to_schema(), out=Schema.to_schema())
        async def static_meta_coroutine(
            df1: DataFrame[Schema],
        ) -> DataFrame[Schema]:
            return df1

    class SomeClass(metaclass=Meta):
        @check_types
        @check_output(Schema.to_schema())
        @check_input(Schema.to_schema(), "df1")
        @check_io(df1=Schema.to_schema(), out=Schema.to_schema())
        async def regular_coroutine(  # pylint: disable=no-self-use
            self,
            df1: DataFrame[Schema],
        ) -> DataFrame[Schema]:
            return df1

        @classmethod
        @check_types
        @check_output(Schema.to_schema())
        @check_input(Schema.to_schema(), "df1")
        @check_io(df1=Schema.to_schema(), out=Schema.to_schema())
        async def class_coroutine(
            cls, df1: DataFrame[Schema]
        ) -> DataFrame[Schema]:
            return df1

        @staticmethod
        @check_types
        @check_output(Schema.to_schema())
        @check_input(Schema.to_schema())
        @check_io(df1=Schema.to_schema(), out=Schema.to_schema())
        async def static_coroutine(
            df1: DataFrame[Schema],
        ) -> DataFrame[Schema]:
            return df1

    async def check_coros() -> None:
        good_df: DataFrame[Schema] = DataFrame({Schema.col1: [1]})
        bad_df: DataFrame[Schema] = DataFrame({"bad_schema": [1]})
        instance = SomeClass()
        for coro in [
            coroutine,
            instance.regular_coroutine,
            SomeClass.class_coroutine,
            instance.static_coroutine,
            SomeClass.static_coroutine,
            SomeClass.class_meta_coroutine,
            SomeClass.static_meta_coroutine,
            SomeClass.regular_meta_coroutine,
        ]:
            res = await coro(good_df)
            pd.testing.assert_frame_equal(good_df, res)

            with pytest.raises(errors.SchemaError):
                await coro(bad_df)

    event_loop.run_until_complete(check_coros())
