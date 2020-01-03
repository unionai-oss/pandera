import numpy as np
import pandas as pd
import pytest

from pandera import (
    Column, DataFrameSchema, Index, SeriesSchema, Bool, Category, Check,
    DateTime, Float, Int, Object, String, Timedelta, errors)


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
    df = pd.DataFrame(
        {
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


def test_series_schema():
    int_schema = SeriesSchema(
        Int, Check(lambda x: 0 <= x <= 100, element_wise=True))
    assert isinstance(int_schema.validate(
        pd.Series([0, 30, 50, 100])), pd.Series)

    str_schema = SeriesSchema(
        String, Check(lambda s: s.isin(["foo", "bar", "baz"])),
        nullable=True, coerce=True)
    assert isinstance(str_schema.validate(
        pd.Series(["foo", "bar", "baz", None])), pd.Series)
    assert isinstance(str_schema.validate(
        pd.Series(["foo", "bar", "baz", np.nan])), pd.Series)

    # error cases
    for data in [-1, 101, 50.1, "foo"]:
        with pytest.raises(errors.SchemaError):
            int_schema.validate(pd.Series([data]))

    for data in [-1, {"a": 1}, -1.0]:
        with pytest.raises(TypeError):
            int_schema.validate(TypeError)

    non_duplicate_schema = SeriesSchema(
        Int, allow_duplicates=False)
    with pytest.raises(errors.SchemaError):
        non_duplicate_schema.validate(pd.Series([0, 1, 2, 3, 4, 1]))

    # when series name doesn't match schema
    named_schema = SeriesSchema(Int, name="my_series")
    with pytest.raises(
            errors.SchemaError,
            match=r"^Expected .+ to have name"):
        named_schema.validate(pd.Series(range(5), name="your_series"))

    # when series floats are declared to be integer
    with pytest.raises(
            errors.SchemaError,
            match=r"^after dropping null values, expected values in series"):
        SeriesSchema(Int, nullable=True).validate(
            pd.Series([1.1, 2.3, 5.5, np.nan]))

    # when series contains null values when schema is not nullable
    with pytest.raises(
            errors.SchemaError,
            match=r"^non-nullable series .+ contains null values"):
        SeriesSchema(Float, nullable=False).validate(
            pd.Series([1.1, 2.3, 5.5, np.nan]))

    # when series contains null values when schema is not nullable in addition
    # to having the wrong data type
    with pytest.raises(
            errors.SchemaError,
            match=(
                r"^expected series '.+' to have type .+, got .+ and "
                "non-nullable series contains null values")):
        SeriesSchema(Int, nullable=False).validate(
            pd.Series([1.1, 2.3, 5.5, np.nan]))


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


class SeriesGreaterCheck:
    # pylint: disable=too-few-public-methods
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


def series_greater_than_zero(s: pd.Series):
    """Return a bool series indicating whether the elements of s are > 0"""
    return s > 0


def series_greater_than_ten(s: pd.Series):
    """Return a bool series indicating whether the elements of s are > 10"""
    return s > 10


@pytest.mark.parametrize("check_function, should_fail", [
    (lambda s: s > 0, False),
    (lambda s: s > 10, True),
    (series_greater_than_zero, False),
    (series_greater_than_ten, True),
    (SeriesGreaterCheck(lower_bound=0), False),
    (SeriesGreaterCheck(lower_bound=10), True)
])
def test_dataframe_schema_check_function_types(check_function, should_fail):
    schema = DataFrameSchema(
        {
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


def test_nullable_int_in_dataframe():
    df = pd.DataFrame({"column1": [5, 1, np.nan]})
    null_schema = DataFrameSchema({
        "column1": Column(Int, Check(lambda x: x > 0), nullable=True)
    })
    assert isinstance(null_schema.validate(df), pd.DataFrame)

    # test case where column is an object
    df = df.astype({"column1": "object"})
    assert isinstance(null_schema.validate(df), pd.DataFrame)


def test_coerce_dtype_in_dataframe():
    df = pd.DataFrame({
        "column1": [10.0, 20.0, 30.0],
        "column2": ["2018-01-01", "2018-02-01", "2018-03-01"],
        "column3": [1, 2, None],
        "column4": [1., 1., np.nan],
    })
    # specify `coerce` at the Column level
    schema1 = DataFrameSchema({
        "column1": Column(Int, Check(lambda x: x > 0), coerce=True),
        "column2": Column(DateTime, coerce=True),
        "column3": Column(String, coerce=True, nullable=True),
    })
    # specify `coerce` at the DataFrameSchema level
    schema2 = DataFrameSchema({
        "column1": Column(Int, Check(lambda x: x > 0)),
        "column2": Column(DateTime),
        "column3": Column(String, nullable=True),
    }, coerce=True)

    for schema in [schema1, schema2]:
        result = schema.validate(df)
        assert result.column1.dtype == Int.value
        assert result.column2.dtype == DateTime.value
        for _, x in result.column3.iteritems():
            assert pd.isna(x) or isinstance(x, str)

        # make sure that correct error is raised when null values are present
        # in a float column that's coerced to an int
        schema = DataFrameSchema({"column4": Column(Int, coerce=True)})
        with pytest.raises(ValueError):
            schema.validate(df)


def test_coerce_dtype_nullable_str():
    df_nans = pd.DataFrame({
        "col": ["foobar", "foo", "bar", "baz", np.nan, np.nan],
    })
    df_nones = pd.DataFrame({
        "col": ["foobar", "foo", "bar", "baz", None, None],
    })

    with pytest.raises(errors.SchemaError):
        for df in [df_nans, df_nones]:
            DataFrameSchema({
                "col": Column(String, coerce=True, nullable=False)
            }).validate(df)

    schema = DataFrameSchema({
        "col": Column(String, coerce=True, nullable=True)
    })

    for df in [df_nans, df_nones]:
        assert isinstance(schema.validate(df), pd.DataFrame)


def test_no_dtype_dataframe():
    schema = DataFrameSchema({"col": Column(nullable=False)})
    validated_df = schema.validate(pd.DataFrame({"col": [-123.1, -76.3, 1.0]}))
    assert isinstance(validated_df, pd.DataFrame)

    schema = DataFrameSchema({"col": Column(nullable=True)})
    validated_df = schema.validate(pd.DataFrame({"col": [-123.1, None, 1.0]}))
    assert isinstance(validated_df, pd.DataFrame)

    with pytest.raises(errors.SchemaError):
        schema = DataFrameSchema({"col": Column(nullable=False)})
        schema.validate(pd.DataFrame({"col": [-123.1, None, 1.0]}))


def test_no_dtype_series():
    schema = SeriesSchema(nullable=False)
    validated_series = schema.validate(pd.Series([0, 1, 2, 3, 4, 1]))
    assert isinstance(validated_series, pd.Series)

    schema = SeriesSchema(nullable=True)
    validated_series = schema.validate(pd.Series([0, 1, 2, None, 4, 1]))
    assert isinstance(validated_series, pd.Series)

    with pytest.raises(errors.SchemaError):
        schema = SeriesSchema(nullable=False)
        schema.validate(pd.Series([0, 1, 2, None, 4, 1]))


def test_coerce_without_dtype():
    with pytest.raises(errors.SchemaInitError):
        DataFrameSchema({"col": Column(coerce=True)})

    with pytest.raises(errors.SchemaInitError):
        DataFrameSchema({"col": Column()}, coerce=True)


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


def test_dataframe_schema_str_repr():
    schema = DataFrameSchema(
        columns={
            "col1": Column(Int),
            "col2": Column(String),
            "col3": Column(DateTime),
        },
        index=Index(Int, name="my_index"),

    )

    for x in [schema.__str__(), schema.__repr__()]:
        assert isinstance(x, str)
        assert schema.__class__.__name__ in x
        for name in ["col1", "col2", "col3", "my_index"]:
            assert name in x


def test_dataframe_schema_dtype_property():
    schema = DataFrameSchema(
        columns={
            "col1": Column(Int),
            "col2": Column(String),
            "col3": Column(DateTime),
            "col4": Column("uint16"),
        }
    )
    assert schema.dtype == {
        "col1": "int64",
        "col2": "object",
        "col3": "datetime64[ns]",
        "col4": "uint16"
    }

@pytest.mark.parametrize("pandas_dtype, expected", [
    (Bool, "bool"),
    (DateTime, "datetime64[ns]"),
    (Category, "category"),
    (Float, "float64"),
    (Int, "int64"),
    (Object, "object"),
    (String, "object"),
    (Timedelta, "timedelta64[ns]"),
    ("bool", "bool"),
    ("datetime64[ns]", "datetime64[ns]"),
    ("category", "category"),
    ("float64", "float64"),
    ("float64", "float64"),
])
def test_series_schema_dtype_property(pandas_dtype, expected):
    assert SeriesSchema(pandas_dtype).dtype == expected
