"""Some unit tests."""

import numpy as np
import pandas as pd
import pytest

from schema import SchemaError

from pandera import Column, DataFrameSchema, Index, PandasDtype, \
    SeriesSchema, Validator, validate_input, validate_output


def test_series_schema():
    schema = SeriesSchema(
        PandasDtype.Int, Validator(lambda x: 0 <= x <= 100))
    validated_series = schema.validate(pd.Series([0, 30, 50, 100]))
    assert isinstance(validated_series, pd.Series)

    # error cases
    for data in [-1, 101, 50.1, "foo"]:
        with pytest.raises(SchemaError):
            schema.validate(pd.Series([data]))

    for data in [-1, {"a": 1}, -1.0]:
        with pytest.raises(TypeError):
            schema.validate(TypeError)


def test_series_schema_multiple_validators():
    schema = SeriesSchema(
        PandasDtype.Int, [
            Validator(lambda x: 0 <= x <= 50),
            Validator(lambda s: 21 in s.values, element_wise=False)])
    validated_series = schema.validate(pd.Series([1, 5, 21, 50]))
    assert isinstance(validated_series, pd.Series)

    # raise error if any of the validators fails
    with pytest.raises(SchemaError):
        schema.validate(pd.Series([1, 5, 20, 50]))


def test_dataframe_schema():
    schema = DataFrameSchema(
        [
            Column("a", PandasDtype.Int, Validator(lambda x: x > 0)),
            Column("b", PandasDtype.Float, Validator(lambda x: 0 <= x <= 10)),
            Column("c", PandasDtype.String,
                   Validator(lambda x: set(x) == {"x", "y", "z"},
                             element_wise=False)),
            Column("d", PandasDtype.Bool,
                   Validator(lambda x: x.mean() > 0.5, element_wise=False)),
            Column("e", PandasDtype.Category,
                   Validator(lambda x: set(x) == {"c1", "c2", "c3"},
                             element_wise=False)),
            Column("f", PandasDtype.Object,
                   Validator(lambda x: x.isin([(1,), (2,), (3,)]),
                             element_wise=False)),
            Column("g", PandasDtype.DateTime,
                   Validator(lambda x: x >= pd.Timestamp("2015-01-01"))),
            Column("i", PandasDtype.Timedelta,
                   Validator(lambda x: x < pd.Timedelta(10, unit="D")))
        ])
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
    with pytest.raises(SchemaError):
        schema.validate(df.drop("a", axis=1))


def test_index_schema():
    schema = Index(
        PandasDtype.Int, [
            Validator(lambda x: 1 <= x <= 12),
            Validator(lambda index: index.mean() > 1, element_wise=False)],
        to_series=True)
    df = pd.DataFrame(index=pd.Index(range(1, 11), dtype="int64"))
    schema(df)


def test_validate_decorators():
    in_schema = DataFrameSchema(
        [
            Column("a", PandasDtype.Int, [
                Validator(lambda x: x >= 1),
                Validator(lambda s: s.mean() > 0, element_wise=False)]),
            Column("b", PandasDtype.String,
                   Validator(lambda x: x in ["x", "y", "z"])),
            Column("c", PandasDtype.DateTime,
                   Validator(lambda x: pd.Timestamp("2018-01-01") <= x)),
            Column("d", PandasDtype.Float,
                   Validator(lambda x: np.isnan(x) or x < 3), nullable=True)
        ],
        transformer=lambda df: df.assign(e="foo")
    )
    out_schema = DataFrameSchema(
        [
            Column("e", PandasDtype.String,
                   Validator(lambda s: s == "foo", element_wise=False)),
            Column("f", PandasDtype.String,
                   Validator(lambda x: x in ["a", "b"]))])

    # case 1: simplest path test - df is first argument and function returns
    # single dataframe as output.
    @validate_input(in_schema)
    @validate_output(out_schema)
    def test_func1(dataframe, x):
        return dataframe.assign(f=["a", "b", "a"])

    # case 2: input and output validation using positional arguments
    @validate_input(in_schema, 1)
    @validate_output(out_schema, 0)
    def test_func2(x, dataframe):
        return dataframe.assign(f=["a", "b", "a"]), x

    # case 3: dataframe to validate is called as a keyword argument and the
    # output is in a dictionary
    @validate_input(in_schema, "in_dataframe")
    @validate_output(out_schema, "out_dataframe")
    def test_func3(x, in_dataframe=None):
        return {
            "x": x,
            "out_dataframe": in_dataframe.assign(f=["a", "b", "a"]),
        }

    # case 4: dataframe is a positional argument but the obj_getter in the
    # validate_input decorator refers to the argument name of the dataframe
    @validate_input(in_schema, "dataframe")
    @validate_output(out_schema)
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
    # argument, the validate_input decorator should still be able to handle
    # it.
    result = test_func3("foo", df)
    assert result["x"] == "foo"
    assert isinstance(df, pd.DataFrame)

    df = test_func4("foo", df)
    assert x == "foo"
    assert isinstance(df, pd.DataFrame)
