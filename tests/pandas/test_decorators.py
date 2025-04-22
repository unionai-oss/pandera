"""Testing the Decorators that check a functions input or output."""

import pickle
import typing
from asyncio import AbstractEventLoop

import numpy as np
import pandas as pd
import pytest

from pandera.pandas import (
    Check,
    Column,
    DataFrameModel,
    DataFrameSchema,
    DateTime,
    Field,
    Float,
    Int,
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
    # python 3.8+
    from typing import Literal  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    from typing_extensions import Literal  # type: ignore[assignment]


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


def test_check_decorator_coercion() -> None:
    """Test that check decorators correctly coerce input/output data."""

    in_schema = DataFrameSchema({"column1": Column(int, coerce=True)})
    out_schema = DataFrameSchema({"column2": Column(float, coerce=True)})

    @check_input(in_schema)
    @check_output(out_schema)
    def test_func_io(df):
        return df.assign(column2=10)

    @check_input(in_schema)
    @check_output(out_schema, obj_getter=1)
    def test_func_out_tuple_obj_getter(df):
        return None, df.assign(column2=10)

    @check_input(in_schema)
    @check_output(out_schema, obj_getter=1)
    def test_func_out_list_obj_getter(df):
        return None, df.assign(column2=10)

    @check_input(in_schema)
    @check_output(out_schema, obj_getter="key")
    def test_func_out_dict_obj_getter(df):
        return {"key": df.assign(column2=10)}

    cases: typing.Iterable[
        typing.Tuple[typing.Callable, typing.Union[int, str, None]]
    ] = [
        (test_func_io, None),
        (test_func_out_tuple_obj_getter, 1),
        (test_func_out_list_obj_getter, 1),
        (test_func_out_dict_obj_getter, "key"),
    ]
    for fn, key in cases:
        out = fn(pd.DataFrame({"column1": ["1", "2", "3"]}))
        if key is not None:
            out = out[key]
        assert out.dtypes["column1"] == "int64"
        assert out.dtypes["column2"] == "float64"


def test_check_output_coercion_error() -> None:
    """Test that check_output raises ValueError when obj_getter is callable."""

    with pytest.raises(
        ValueError,
        match="Cannot use callable obj_getter when the schema uses coercion",
    ):

        @check_output(
            DataFrameSchema({"column2": Column(float, coerce=True)}),
            obj_getter=lambda x: x[0]["key"],
        )
        def test_func(df):  # pylint: disable=unused-argument
            ...


def test_check_instance_method_decorator_error() -> None:
    """Test error message on methods."""

    # pylint: disable-next=missing-class-docstring
    class TestClass:
        @check_input(DataFrameSchema({"column1": Column(Int)}))
        def test_method(self, df):
            # pylint: disable=missing-function-docstring
            return df

    with pytest.raises(
        errors.SchemaError,
        match=r"^error in check_input decorator of function 'test_method'",
    ):
        test_instance = TestClass()
        test_instance.test_method(pd.DataFrame({"column2": ["a", "b", "c"]}))


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


class DfModel(DataFrameModel):
    col: int


# pylint: disable=unused-argument
@check_input(DfModel.to_schema())
def fn_with_check_input(data: DataFrame[DfModel], *, kwarg: bool = False):
    return data


def test_check_input_on_fn_with_kwarg():
    """
    That that a check_input correctly validates a function where the first arg
    is the dataframe and the function has other kwargs.
    """
    df = pd.DataFrame({"col": [1]})
    fn_with_check_input(df, kwarg=True)


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
        # pylint: disable=undefined-loop-variable
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

    def _assert_equals(actual, expect):
        if isinstance(actual, pd.Series):
            assert (actual == expect).all()
        elif isinstance(actual, pd.DataFrame):
            assert (actual == expect).all(axis=None)
        else:
            assert actual == expect

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
            _assert_equals(result, out)
        if isinstance(result, pd.DataFrame):
            _assert_equals(result, out)
        else:
            for _result, _out in zip(result, out):  # type: ignore
                _assert_equals(_result, _out)

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
class OnlyZeroesSchema(
    DataFrameModel
):  # pylint:disable=too-few-public-methods
    """Schema with a single column containing zeroes."""

    a: Series[int] = Field(eq=0)


class OnlyOnesSchema(DataFrameModel):  # pylint:disable=too-few-public-methods
    """Schema with a single column containing ones."""

    a: Series[int] = Field(eq=1)


def test_check_types_arguments() -> None:
    """Test that check_types forwards key-words arguments to validate."""
    df = pd.DataFrame({"a": [0, 0]})

    @check_types()
    def transform_empty_parenthesis(
        df: DataFrame[OnlyZeroesSchema],
    ) -> DataFrame[OnlyZeroesSchema]:  # pylint: disable=unused-argument
        return df

    transform_empty_parenthesis(df)  # type: ignore

    @check_types(head=1)
    def transform_head(
        df: DataFrame[OnlyZeroesSchema],  # pylint: disable=unused-argument
    ) -> DataFrame[OnlyZeroesSchema]:
        return pd.DataFrame({"a": [0, 0]})  # type: ignore

    transform_head(df)  # type: ignore

    @check_types(tail=1)
    def transform_tail(
        df: DataFrame[OnlyZeroesSchema],  # pylint: disable=unused-argument
    ) -> DataFrame[OnlyZeroesSchema]:
        return pd.DataFrame({"a": [1, 0]})  # type: ignore

    transform_tail(df)  # type: ignore

    @check_types(lazy=True)
    def transform_lazy(
        df: DataFrame[OnlyZeroesSchema],  # pylint: disable=unused-argument
    ) -> DataFrame[OnlyZeroesSchema]:
        return pd.DataFrame({"a": [1, 1]})  # type: ignore

    with pytest.raises(errors.SchemaErrors, match=r"DATA"):
        transform_lazy(df)  # type: ignore


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
    pd.testing.assert_frame_equal(transform(df, 2), df)  # type: ignore


# required to be globals:
# see https://pydantic-docs.helpmanual.io/usage/postponed_annotations/
class InSchema(DataFrameModel):  # pylint:disable=too-few-public-methods
    """Test schema used as input."""

    a: Series[int]
    idx: Index[str]

    class Config:  # pylint: disable=too-few-public-methods
        """Set coerce."""

        coerce = True


class DerivedOutSchema(InSchema):
    """Test schema derived from InSchema."""

    b: Series[int]


class OutSchema(DataFrameModel):  # pylint: disable=too-few-public-methods
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
    transform(correct, df_2=correct)  # type: ignore

    wrong = pd.DataFrame({"b": [1]})
    with pytest.raises(
        errors.SchemaError, match="column 'a' not in dataframe"
    ):
        transform(correct, wrong)  # type: ignore


def test_check_types_error_input() -> None:
    """Test that check_types raises an error when the input is not correct."""

    @check_types
    def transform(df: DataFrame[InSchema]):
        return df

    df = pd.DataFrame({"b": [1]})
    with pytest.raises(
        errors.SchemaError, match="column 'a' not in dataframe"
    ):
        transform(df)  # type: ignore

    try:
        transform(df)  # type: ignore
    except errors.SchemaError as exc:
        assert exc.schema == InSchema.to_schema()


def test_check_types_error_output() -> None:
    """Test that check_types raises an error when the output is not correct."""

    df = pd.DataFrame({"a": [1]}, index=["1"])

    @check_types
    def transform_derived(
        df: DataFrame[InSchema],
    ) -> DataFrame[DerivedOutSchema]:
        return df  # type: ignore

    with pytest.raises(
        errors.SchemaError, match="column 'b' not in dataframe"
    ):
        transform_derived(df)  # type: ignore

    try:
        transform_derived(df)  # type: ignore
    except errors.SchemaError as exc:
        assert exc.schema == DerivedOutSchema.to_schema()

    df = pd.DataFrame({"a": [1]}, index=["1"])

    @check_types
    def transform(df: DataFrame[InSchema]) -> DataFrame[OutSchema]:
        return df  # type: ignore

    with pytest.raises(
        errors.SchemaError, match="column 'b' not in dataframe"
    ):
        transform(df)  # type: ignore

    try:
        transform(df)  # type: ignore
    except errors.SchemaError as exc:
        assert exc.schema == OutSchema.to_schema()


def test_check_types_optional_out() -> None:
    """Test the check_types behaviour when the output schema is Optional."""

    @check_types
    def optional_derived_out(
        df: DataFrame[InSchema],  # pylint: disable=unused-argument
    ) -> typing.Optional[DataFrame[DerivedOutSchema]]:
        return None

    df = pd.DataFrame({"a": [1]}, index=["1"])
    assert optional_derived_out(df) is None  # type: ignore

    @check_types
    def optional_out(
        df: DataFrame[InSchema],  # pylint: disable=unused-argument
    ) -> typing.Optional[DataFrame[OutSchema]]:
        return None

    df = pd.DataFrame({"a": [1]}, index=["1"])
    assert optional_out(df) is None  # type: ignore


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

    df = transform_in(pd.DataFrame({"a": ["1"]}, index=["1"]))  # type: ignore
    expected = InSchema.to_schema().columns["a"].dtype
    assert Engine.dtype(df["a"].dtype) == expected

    @check_types()
    def transform_out() -> DataFrame[OutSchema]:
        # OutSchema.b should be coerced to an integer.
        return pd.DataFrame({"b": ["1"]})  # type: ignore

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
        arg_type = Literal[example]  # type: ignore

        @check_types
        def transform_with_literal(
            df: DataFrame[InSchema],
            # pylint: disable=unused-argument,cell-var-from-loop
            arg: arg_type,  # type: ignore
        ) -> DataFrame[OutSchema]:
            return df.assign(b=100)  # type: ignore

        df = pd.DataFrame({"a": [1]}, index=["a"])
        invalid_df = pd.DataFrame()

        transform_with_literal(df, example)
        with pytest.raises(errors.SchemaError):
            transform_with_literal(invalid_df, example)


def test_check_types_method_args() -> None:
    """Test that @check_types works with positional and keyword args in methods,
    classmethods and staticmethods.
    """
    # pylint: disable=unused-argument,missing-class-docstring,too-few-public-methods,missing-function-docstring

    class SchemaIn1(DataFrameModel):
        col1: Series[int]

        class Config:
            strict = True

    class SchemaIn2(DataFrameModel):
        col2: Series[int]

        class Config:
            strict = True

    class SchemaOut(DataFrameModel):
        col3: Series[int]

        class Config:
            strict = True

    in1: DataFrame[SchemaIn1] = DataFrame({SchemaIn1.col1: [1]})
    in2: DataFrame[SchemaIn2] = DataFrame({SchemaIn2.col2: [2]})
    out: DataFrame[SchemaOut] = DataFrame({SchemaOut.col3: [3]})

    class SomeClass:
        @check_types
        def regular_method(
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
        instance.regular_method(in2, in1)  # type: ignore
    with pytest.raises(errors.SchemaError):
        instance.regular_method(in2, df2=in1)  # type: ignore
    with pytest.raises(errors.SchemaError):
        instance.regular_method(df1=in2, df2=in1)  # type: ignore

    pd.testing.assert_frame_equal(out, SomeClass.class_method(in1, in2))
    pd.testing.assert_frame_equal(out, SomeClass.class_method(in1, df2=in2))
    pd.testing.assert_frame_equal(
        out, SomeClass.class_method(df1=in1, df2=in2)
    )

    with pytest.raises(errors.SchemaError):
        instance.class_method(in2, in1)  # type: ignore
    with pytest.raises(errors.SchemaError):
        instance.class_method(in2, df2=in1)  # type: ignore
    with pytest.raises(errors.SchemaError):
        instance.class_method(df1=in2, df2=in1)  # type: ignore

    pd.testing.assert_frame_equal(out, instance.static_method(in1, in2))
    pd.testing.assert_frame_equal(out, SomeClass.static_method(in1, in2))
    pd.testing.assert_frame_equal(out, instance.static_method(in1, df2=in2))
    pd.testing.assert_frame_equal(out, SomeClass.static_method(in1, df2=in2))
    pd.testing.assert_frame_equal(
        out, instance.static_method(df1=in1, df2=in2)
    )

    with pytest.raises(errors.SchemaError):
        instance.static_method(in2, in1)  # type: ignore
    with pytest.raises(errors.SchemaError):
        instance.static_method(in2, df2=in1)  # type: ignore
    with pytest.raises(errors.SchemaError):
        instance.static_method(df1=in2, df2=in1)  # type: ignore


def test_check_types_union_args() -> None:
    """Test that the @check_types decorator works with
    typing.Union[pandera.typing.DataFrame[S1], pandera.typing.DataFrame[S2]] type inputs/outputs
    """

    @check_types
    def validate_union(
        df: typing.Union[
            DataFrame[OnlyZeroesSchema],
            DataFrame[OnlyOnesSchema],
        ],
    ) -> typing.Union[DataFrame[OnlyZeroesSchema], DataFrame[OnlyOnesSchema]]:
        return df

    validate_union(pd.DataFrame({"a": [0, 0]}))  # type: ignore [arg-type]
    validate_union(pd.DataFrame({"a": [1, 1]}))  # type: ignore [arg-type]

    with pytest.raises(errors.SchemaErrors):
        validate_union(pd.DataFrame({"a": [0, 1]}))  # type: ignore [arg-type]
    with pytest.raises(errors.SchemaErrors):
        validate_union(pd.DataFrame({"a": [2, 2]}))  # type: ignore [arg-type]

    @check_types
    def validate_union_wrong_outputs(
        df: typing.Union[
            DataFrame[OnlyZeroesSchema], DataFrame[OnlyOnesSchema]
        ],
    ) -> typing.Union[DataFrame[OnlyZeroesSchema], DataFrame[OnlyOnesSchema]]:
        new_df = df.copy()
        new_df["a"] = [0, 1]
        return new_df  # type: ignore [return-value]

    with pytest.raises(errors.SchemaErrors):
        validate_union_wrong_outputs(pd.DataFrame({"a": [0, 0]}))  # type: ignore [arg-type]


def test_check_types_non_dataframes() -> None:
    """Test to skip check_types for non-dataframes"""

    @check_types
    def only_int_type(val: int) -> int:
        return val

    @check_types
    def union_int_str_types(
        val: typing.Union[int, str],
    ) -> typing.Union[int, str]:
        return val

    only_int_type(1)
    int_val = union_int_str_types(2)
    str_val = union_int_str_types("2")
    assert isinstance(int_val, int)
    assert isinstance(str_val, str)

    @check_types(with_pydantic=True)
    def union_df_int_types_pydantic_check(
        val: typing.Union[DataFrame[OnlyZeroesSchema], int],
    ) -> typing.Union[DataFrame[OnlyZeroesSchema], int]:
        return val

    union_df_int_types_pydantic_check(pd.DataFrame({"a": [0, 0]}))  # type: ignore [arg-type]
    int_val_pydantic = union_df_int_types_pydantic_check(5)
    str_val_pydantic = union_df_int_types_pydantic_check("5")  # type: ignore[arg-type]
    assert isinstance(int_val_pydantic, int)
    assert isinstance(str_val_pydantic, int)


def test_check_types_star_args() -> None:
    """Test to check_types for functions with *args arguments"""

    @check_types
    def get_len_star_args__int(
        # pylint: disable=unused-argument
        arg1: int,
        *args: int,
    ) -> int:
        return len(args)

    @check_types
    def get_len_star_args__dataframe(
        # pylint: disable=unused-argument
        arg1: DataFrame[InSchema],
        *args: DataFrame[InSchema],
    ) -> int:
        return len(args)

    in_1 = pd.DataFrame({"a": [1]}, index=["1"])
    in_2 = pd.DataFrame({"a": [1]}, index=["1"])
    in_3 = pd.DataFrame({"a": [1]}, index=["1"])
    in_4_error = pd.DataFrame({"b": [1]}, index=["1"])

    assert get_len_star_args__int(1, 2, 3) == 2
    assert get_len_star_args__dataframe(in_1, in_2) == 1
    assert get_len_star_args__dataframe(in_1, in_2, in_3) == 2

    with pytest.raises(
        errors.SchemaError, match="column 'a' not in dataframe"
    ):
        get_len_star_args__dataframe(in_1, in_2, in_4_error)


def test_check_types_star_kwargs() -> None:
    """Test to check_types for functions with **kwargs arguments"""

    @check_types
    def get_star_kwargs_keys_int(
        # pylint: disable=unused-argument
        kwarg1: int = 1,
        **kwargs: int,
    ) -> typing.List[str]:
        return list(kwargs.keys())

    @check_types
    def get_star_kwargs_keys_dataframe(
        # pylint: disable=unused-argument
        kwarg1: typing.Optional[DataFrame[InSchema]] = None,
        **kwargs: DataFrame[InSchema],
    ) -> typing.List[str]:
        return list(kwargs.keys())

    in_1 = pd.DataFrame({"a": [1]}, index=["1"])
    in_2 = pd.DataFrame({"a": [1]}, index=["1"])
    in_3 = pd.DataFrame({"a": [1]}, index=["1"])
    in_4_error = pd.DataFrame({"b": [1]}, index=["1"])

    int_kwargs_keys = get_star_kwargs_keys_int(kwarg1=1, kwarg2=2, kwarg3=3)
    df_kwargs_keys_1 = get_star_kwargs_keys_dataframe(
        kwarg1=in_1,
        kwarg2=in_2,
    )
    df_kwargs_keys_2 = get_star_kwargs_keys_dataframe(
        kwarg1=in_1, kwarg2=in_2, kwarg3=in_3
    )

    assert int_kwargs_keys == ["kwarg2", "kwarg3"]
    assert df_kwargs_keys_1 == ["kwarg2"]
    assert df_kwargs_keys_2 == ["kwarg2", "kwarg3"]

    with pytest.raises(
        errors.SchemaError, match="column 'a' not in dataframe"
    ):
        get_star_kwargs_keys_dataframe(
            kwarg1=in_1, kwarg2=in_2, kwarg3=in_4_error
        )


def test_check_types_star_args_kwargs() -> None:
    """Test to check_types for functions with both *args and **kwargs"""

    @check_types
    def star_args_kwargs(
        arg1: DataFrame[InSchema],
        *args: DataFrame[InSchema],
        kwarg1: DataFrame[InSchema],
        **kwargs: DataFrame[InSchema],
    ):
        return arg1, args, kwarg1, kwargs

    in_1 = pd.DataFrame({"a": [1]}, index=["1"])
    in_2 = pd.DataFrame({"a": [1]}, index=["1"])
    in_3 = pd.DataFrame({"a": [1]}, index=["1"])

    expected_arg = in_1
    expected_star_args = (in_2, in_3)
    expected_kwarg = in_1
    expected_star_kwargs = {"kwarg2": in_2, "kwarg3": in_3}

    arg, star_args, kwarg, star_kwargs = star_args_kwargs(
        in_1, in_2, in_3, kwarg1=in_1, kwarg2=in_2, kwarg3=in_3
    )

    pd.testing.assert_frame_equal(expected_arg, arg)
    pd.testing.assert_frame_equal(expected_kwarg, kwarg)

    for expected, actual in zip(expected_star_args, star_args):
        pd.testing.assert_frame_equal(expected, actual)

    for expected, actual in zip(
        expected_star_kwargs.values(), star_kwargs.values()
    ):
        pd.testing.assert_frame_equal(expected, actual)


def test_coroutines(event_loop: AbstractEventLoop) -> None:
    # pylint: disable=missing-class-docstring,too-few-public-methods,missing-function-docstring
    class Schema(DataFrameModel):
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
        async def regular_meta_coroutine(
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
        async def regular_coroutine(
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


class Schema(DataFrameModel):
    column1: Series[int]
    column2: Series[float]
    column3: Series[str]


@check_types
def process_data_and_check_types(df: DataFrame[Schema]) -> DataFrame[Schema]:
    # Example processing: add a new column
    return df


def test_pickle_decorated_function(tmp_path):
    path = tmp_path / "tmp.pkl"

    with path.open("wb") as f:
        pickle.dump(process_data_and_check_types, f)

    with path.open("rb") as f:
        _process_data_and_check_types = pickle.load(f)

    # pylint: disable=comparison-with-callable
    assert process_data_and_check_types == _process_data_and_check_types
