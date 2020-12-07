"""Testing creation and manipulation of DataFrameSchema objects."""
# pylint: disable=too-many-lines,redefined-outer-name

import copy
from functools import partial

import numpy as np
import pandas as pd
import pytest

from pandera import (
    STRING,
    Bool,
    Category,
    Check,
    Column,
    DataFrameSchema,
    DateTime,
    Float,
    Index,
    Int,
    MultiIndex,
    Object,
    PandasDtype,
    SeriesSchema,
    String,
    Timedelta,
    errors,
)
from pandera.dtypes import LEGACY_PANDAS
from pandera.schemas import SeriesSchemaBase
from tests.test_dtypes import TESTABLE_DTYPES


def test_dataframe_schema():
    """Tests the Checking of a DataFrame that has a wide variety of types and
    conditions. Tests include: when the Schema works, when a column is dropped,
    and when a columns values change its type.
    """
    schema = DataFrameSchema(
        {
            "a": Column(Int, Check(lambda x: x > 0, element_wise=True)),
            "b": Column(
                Float, Check(lambda x: 0 <= x <= 10, element_wise=True)
            ),
            "c": Column(String, Check(lambda x: set(x) == {"x", "y", "z"})),
            "d": Column(Bool, Check(lambda x: x.mean() > 0.5)),
            "e": Column(
                Category, Check(lambda x: set(x) == {"c1", "c2", "c3"})
            ),
            "f": Column(Object, Check(lambda x: x.isin([(1,), (2,), (3,)]))),
            "g": Column(
                DateTime,
                Check(
                    lambda x: x >= pd.Timestamp("2015-01-01"),
                    element_wise=True,
                ),
            ),
            "i": Column(
                Timedelta,
                Check(
                    lambda x: x < pd.Timedelta(10, unit="D"), element_wise=True
                ),
            ),
        }
    )
    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [1.1, 2.5, 9.9],
            "c": ["z", "y", "x"],
            "d": [True, True, False],
            "e": pd.Series(["c2", "c1", "c3"], dtype="category"),
            "f": [(3,), (2,), (1,)],
            "g": [
                pd.Timestamp("2015-02-01"),
                pd.Timestamp("2015-02-02"),
                pd.Timestamp("2015-02-03"),
            ],
            "i": [
                pd.Timedelta(1, unit="D"),
                pd.Timedelta(5, unit="D"),
                pd.Timedelta(9, unit="D"),
            ],
        }
    )
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
    """
    Checks if strict=True whether a schema error is raised because 'a' is
    not present in the dataframe.
    """
    schema = DataFrameSchema({"a": Column(Int, nullable=True)}, strict=True)
    df = pd.DataFrame({"b": [1, 2, 3]})
    with pytest.raises(errors.SchemaError):
        schema.validate(df)


def test_dataframe_schema_strict_regex():
    """Test that strict dataframe schema checks for regex matches."""
    schema = DataFrameSchema(
        {"foo_*": Column(Int, regex=True)},
        strict=True,
    )
    df = pd.DataFrame({"foo_%d" % i: range(10) for i in range(5)})

    assert isinstance(schema.validate(df), pd.DataFrame)

    # Raise a SchemaError if schema is strict and a regex pattern yields
    # no matches
    with pytest.raises(errors.SchemaError):
        schema.validate(
            pd.DataFrame({"bar_%d" % i: range(10) for i in range(5)})
        )


def test_dataframe_pandas_dtype_coerce():
    """
    Test that pandas dtype specified at the dataframe level overrides
    column data types.
    """
    schema = DataFrameSchema(
        columns={f"column_{i}": Column(float) for i in range(5)},
        pandas_dtype=int,
        coerce=True,
    )

    df = pd.DataFrame({f"column_{i}": range(10) for i in range(5)}).astype(
        float
    )
    assert (schema(df).dtypes == Int.str_alias).all()

    # test that pandas_dtype in columns are preserved
    for col in schema.columns.values():
        assert col.pandas_dtype is float

    # raises SchemeError if dataframe can't be coerced
    with pytest.raises(errors.SchemaErrors):
        schema.coerce_dtype(pd.DataFrame({"foo": list("abcdef")}))

    # raises SchemaErrors on lazy validation
    with pytest.raises(errors.SchemaErrors):
        schema(pd.DataFrame({"foo": list("abcdef")}), lazy=True)

    # test that original dataframe dtypes are preserved
    assert (df.dtypes == Float.str_alias).all()

    # test case where pandas_dtype is string
    schema.pandas_dtype = str
    assert (schema(df).dtypes == "object").all()

    schema.pandas_dtype = PandasDtype.String
    assert (schema(df).dtypes == "object").all()

    # raises ValueError if _coerce_dtype is called when pandas_dtype is None
    schema.pandas_dtype = None
    with pytest.raises(ValueError):
        schema._coerce_dtype(df)


def test_dataframe_coerce_regex():
    """Test dataframe pandas dtype coercion for regex columns"""
    schema = DataFrameSchema(
        columns={"column_": Column(float, regex=True, required=False)},
        pandas_dtype=int,
        coerce=True,
    )

    no_match_df = pd.DataFrame({"foo": [1, 2, 3]})
    match_valid_df = pd.DataFrame(
        {
            "column_1": [1, 2, 3],
            "column_2": ["1", "2", "3"],
        }
    )

    schema(no_match_df)
    schema(match_valid_df)

    # if the regex column is required, no matches should raise an error
    schema_required = schema.update_column("column_", required=True)
    with pytest.raises(
        errors.SchemaError, match="Column regex name='column_' did not match"
    ):
        schema_required(no_match_df)


def test_dataframe_reset_column_name():
    """Test resetting column name at DataFrameSchema init on named column."""
    with pytest.warns(UserWarning):
        DataFrameSchema(columns={"new_name": Column(name="old_name")})


def test_series_schema():
    """Tests that a SeriesSchema Check behaves as expected for integers and
    strings. Tests error cases for types, duplicates, name errors, and issues
    around float and integer handling of nulls"""

    SeriesSchema("int").validate(pd.Series([1, 2, 3]))

    int_schema = SeriesSchema(
        Int, Check(lambda x: 0 <= x <= 100, element_wise=True)
    )
    assert isinstance(
        int_schema.validate(pd.Series([0, 30, 50, 100])), pd.Series
    )

    str_schema = SeriesSchema(
        String,
        Check(lambda s: s.isin(["foo", "bar", "baz"])),
        nullable=True,
        coerce=True,
    )
    assert isinstance(
        str_schema.validate(pd.Series(["foo", "bar", "baz", None])), pd.Series
    )
    assert isinstance(
        str_schema.validate(pd.Series(["foo", "bar", "baz", np.nan])),
        pd.Series,
    )

    # error cases
    for data in [-1, 101, 50.1, "foo"]:
        with pytest.raises(errors.SchemaError):
            int_schema.validate(pd.Series([data]))

    for data in [-1, {"a": 1}, -1.0]:
        with pytest.raises(TypeError):
            int_schema.validate(TypeError)

    non_duplicate_schema = SeriesSchema(Int, allow_duplicates=False)
    with pytest.raises(errors.SchemaError):
        non_duplicate_schema.validate(pd.Series([0, 1, 2, 3, 4, 1]))

    # when series name doesn't match schema
    named_schema = SeriesSchema(Int, name="my_series")
    with pytest.raises(errors.SchemaError, match=r"^Expected .+ to have name"):
        named_schema.validate(pd.Series(range(5), name="your_series"))

    # when series floats are declared to be integer
    with pytest.raises(
        errors.SchemaError,
        match=r"^after dropping null values, expected values in series",
    ):
        SeriesSchema(Int, nullable=True).validate(
            pd.Series([1.1, 2.3, 5.5, np.nan])
        )

    # when series contains null values when schema is not nullable
    with pytest.raises(
        errors.SchemaError,
        match=r"^non-nullable series .+ contains null values",
    ):
        SeriesSchema(Float, nullable=False).validate(
            pd.Series([1.1, 2.3, 5.5, np.nan])
        )

    # when series can't be coerced
    with pytest.raises(
        errors.SchemaError,
        match="Error while coercing",
    ):
        SeriesSchema(Float, coerce=True).validate(pd.Series(list("abcdefg")))


def test_series_schema_checks():
    """Test SeriesSchema check property."""
    series_schema_no_checks = SeriesSchema()
    series_schema_one_check = SeriesSchema(checks=Check.eq(0))
    series_schema_multiple_checks = SeriesSchema(
        checks=[Check.gt(0), Check.lt(100)]
    )

    for schema in [
        series_schema_no_checks,
        series_schema_one_check,
        series_schema_multiple_checks,
    ]:
        assert isinstance(schema.checks, list)

    assert len(series_schema_no_checks.checks) == 0
    assert len(series_schema_one_check.checks) == 1
    assert len(series_schema_multiple_checks.checks) == 2


def test_series_schema_multiple_validators():
    """Tests how multiple Checks on a Series Schema are handled both
    successfully and when errors are expected."""
    schema = SeriesSchema(
        Int,
        [
            Check(lambda x: 0 <= x <= 50, element_wise=True),
            Check(lambda s: (s == 21).any()),
        ],
    )
    validated_series = schema.validate(pd.Series([1, 5, 21, 50]))
    assert isinstance(validated_series, pd.Series)

    # raise error if any of the validators fails
    with pytest.raises(errors.SchemaError):
        schema.validate(pd.Series([1, 5, 20, 50]))


@pytest.mark.parametrize("coerce", [True, False])
def test_series_schema_with_index(coerce):
    """Test SeriesSchema with Index and MultiIndex components."""
    schema_with_index = SeriesSchema(
        pandas_dtype=Int,
        index=Index(Int, coerce=coerce),
    )
    validated_series = schema_with_index(pd.Series([1, 2, 3], index=[1, 2, 3]))
    assert isinstance(validated_series, pd.Series)

    schema_with_multiindex = SeriesSchema(
        pandas_dtype=Int,
        index=MultiIndex(
            [
                Index(Int, coerce=coerce),
                Index(String, coerce=coerce),
            ]
        ),
    )
    multi_index = pd.MultiIndex.from_arrays(
        [[0, 1, 2], ["foo", "bar", "foo"]],
    )
    validated_series_multiindex = schema_with_multiindex(
        pd.Series([1, 2, 3], index=multi_index)
    )
    assert isinstance(validated_series_multiindex, pd.Series)
    assert (validated_series_multiindex.index == multi_index).all()


class SeriesGreaterCheck:
    # pylint: disable=too-few-public-methods
    """Class creating callable objects to check if series elements exceed a
    lower bound.
    """

    def __init__(self, lower_bound):
        self.lower_bound = lower_bound

    def __call__(self, series: pd.Series):
        """Check if the elements of s are > lower_bound.

        :returns Series with bool elements
        """
        return series > self.lower_bound


def series_greater_than_zero(series: pd.Series):
    """Return a bool series indicating whether the elements of s are > 0"""
    return series > 0


def series_greater_than_ten(series: pd.Series):
    """Return a bool series indicating whether the elements of s are > 10"""
    return series > 10


@pytest.mark.parametrize(
    "check_function, should_fail",
    [
        (lambda s: s > 0, False),
        (lambda s: s > 10, True),
        (series_greater_than_zero, False),
        (series_greater_than_ten, True),
        (SeriesGreaterCheck(lower_bound=0), False),
        (SeriesGreaterCheck(lower_bound=10), True),
    ],
)
def test_dataframe_schema_check_function_types(check_function, should_fail):
    """Tests a DataFrameSchema against a variety of Check conditions."""
    schema = DataFrameSchema(
        {
            "a": Column(Int, Check(check_function, element_wise=False)),
            "b": Column(Float, Check(check_function, element_wise=False)),
        }
    )
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1.1, 2.5, 9.9]})
    if should_fail:
        with pytest.raises(errors.SchemaError):
            schema.validate(df)
    else:
        schema.validate(df)


def test_nullable_int_in_dataframe():
    """Tests handling of nullability when datatype is integers."""
    df = pd.DataFrame({"column1": [5, 1, np.nan]})
    null_schema = DataFrameSchema(
        {"column1": Column(Int, Check(lambda x: x > 0), nullable=True)}
    )
    assert isinstance(null_schema.validate(df), pd.DataFrame)

    # test case where column is an object
    df = df.astype({"column1": "object"})
    assert isinstance(null_schema.validate(df), pd.DataFrame)


def test_coerce_dtype_in_dataframe():
    """Tests coercions of datatypes, especially regarding nullable integers."""
    df = pd.DataFrame(
        {
            "column1": [10.0, 20.0, 30.0],
            "column2": ["2018-01-01", "2018-02-01", "2018-03-01"],
            "column3": [1, 2, None],
            "column4": [1.0, 1.0, np.nan],
        }
    )
    # specify `coerce` at the Column level
    schema1 = DataFrameSchema(
        {
            "column1": Column(Int, Check(lambda x: x > 0), coerce=True),
            "column2": Column(DateTime, coerce=True),
            "column3": Column(String, coerce=True, nullable=True),
        }
    )
    # specify `coerce` at the DataFrameSchema level
    schema2 = DataFrameSchema(
        {
            "column1": Column(Int, Check(lambda x: x > 0)),
            "column2": Column(DateTime),
            "column3": Column(String, nullable=True),
        },
        coerce=True,
    )

    for schema in [schema1, schema2]:
        result = schema.validate(df)
        assert result.column1.dtype == Int.str_alias
        assert result.column2.dtype == DateTime.str_alias
        for _, x in result.column3.iteritems():
            assert pd.isna(x) or isinstance(x, str)

        # make sure that correct error is raised when null values are present
        # in a float column that's coerced to an int
        schema = DataFrameSchema({"column4": Column(Int, coerce=True)})
        with pytest.raises(
            errors.SchemaError,
            match=r"^Error while coercing .* to type u{0,1}int[0-9]{1,2}: "
            r"Cannot convert non-finite values \(NA or inf\) to integer",
        ):
            schema.validate(df)


def test_coerce_dtype_nullable_str():
    """Tests how null values are handled in string dtypes."""
    # dataframes with columns where the last two values are null
    df_nans = pd.DataFrame(
        {
            "col": ["foobar", "foo", "bar", "baz", np.nan, np.nan],
        }
    )
    df_nones = pd.DataFrame(
        {
            "col": ["foobar", "foo", "bar", "baz", None, None],
        }
    )

    with pytest.raises(errors.SchemaError):
        for df in [df_nans, df_nones]:
            DataFrameSchema(
                {"col": Column(String, coerce=True, nullable=False)}
            ).validate(df)

    schema = DataFrameSchema(
        {"col": Column(String, coerce=True, nullable=True)}
    )

    for df in [df_nans, df_nones]:
        validated_df = schema.validate(df)
        assert isinstance(validated_df, pd.DataFrame)
        assert pd.isna(validated_df["col"].iloc[-1])
        assert pd.isna(validated_df["col"].iloc[-2])
        for i in range(4):
            assert isinstance(validated_df["col"].iloc[i], str)


def test_no_dtype_dataframe():
    """Test how nullability is handled in DataFrameSchemas where no type is
    specified."""
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
    """Test how nullability is handled in SeriesSchemas where no type is
    specified."""
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
    """Test that an error is thrown when a dtype isn't specified and coerce
    is True."""
    with pytest.raises(errors.SchemaInitError):
        DataFrameSchema({"col": Column(coerce=True)})

    with pytest.raises(errors.SchemaInitError):
        DataFrameSchema({"col": Column()}, coerce=True)


def test_required():
    """Tests how a Required Column is handled when it's not included, included
    and then not specified and a second column which is implicitly required
    isn't available."""
    schema = DataFrameSchema(
        {"col1": Column(Int, required=False), "col2": Column(String)}
    )

    df_ok_1 = pd.DataFrame({"col2": ["hello", "world"]})

    df = schema.validate(df_ok_1)
    assert isinstance(df, pd.DataFrame)
    assert len(df.columns) == 1
    assert set(df.columns) == {"col2"}

    df_ok_2 = pd.DataFrame({"col1": [1, 2], "col2": ["hello", "world"]})

    df = schema.validate(df_ok_2)
    assert isinstance(df, pd.DataFrame)
    assert len(df.columns) == 2
    assert set(df.columns) == {"col1", "col2"}

    df_not_ok = pd.DataFrame({"col1": [1, 2]})

    with pytest.raises(Exception):
        schema.validate(df_not_ok)


def test_head_dataframe_schema():
    """Test that schema can validate head of dataframe, returns entire
    dataframe."""

    df = pd.DataFrame(
        {"col1": list(range(0, 100)) + list(range(-1, -1001, -1))}
    )

    schema = DataFrameSchema(
        columns={"col1": Column(Int, Check(lambda s: s >= 0))}
    )

    # Validating with head of 100 should pass
    assert schema.validate(df, head=100).equals(df)
    with pytest.raises(errors.SchemaError):
        schema.validate(df)


def test_tail_dataframe_schema():
    """Checks that validating the tail of a dataframe validates correctly."""
    df = pd.DataFrame(
        {"col1": list(range(0, 100)) + list(range(-1, -1001, -1))}
    )

    schema = DataFrameSchema(
        columns={"col1": Column(Int, Check(lambda s: s < 0))}
    )

    # Validating with tail of 1000 should pass
    assert schema.validate(df, tail=1000).equals(df)
    with pytest.raises(errors.SchemaError):
        schema.validate(df)


def test_sample_dataframe_schema():
    """Test the sample argument of schema.validate."""
    df = pd.DataFrame({"col1": range(1, 1001)})

    # assert all values -1
    schema = DataFrameSchema(
        columns={"col1": Column(Int, Check(lambda s: s == -1))}
    )

    for seed in [11, 123456, 9000, 654]:
        sample_index = df.sample(100, random_state=seed).index
        df.loc[sample_index] = -1
        assert schema.validate(df, sample=100, random_state=seed).equals(df)


def test_dataframe_schema_str_repr():
    """Test the __str__ and __repr__ methods which are used for cleanly
    printing/logging of a DataFrameSchema."""
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
    """Test that schema.dtype returns the matching Column types."""
    schema = DataFrameSchema(
        columns={
            "col1": Column(Int),
            "col2": Column(String),
            "col3": Column(STRING),
            "col4": Column(DateTime),
            "col5": Column("uint16"),
        }
    )
    assert schema.dtype == {
        "col1": "int64",
        "col2": "object",
        "col3": ("object" if LEGACY_PANDAS else "string"),
        "col4": "datetime64[ns]",
        "col5": "uint16",
    }


@pytest.mark.parametrize("pandas_dtype, expected", TESTABLE_DTYPES)
def test_series_schema_dtype_property(pandas_dtype, expected):
    """Tests every type of allowed dtype."""
    assert SeriesSchema(pandas_dtype).dtype == expected


def test_schema_equality_operators():
    """Test the usage of == for DataFrameSchema, SeriesSchema and
    SeriesSchemaBase."""
    df_schema = DataFrameSchema(
        {
            "col1": Column(Int, Check(lambda s: s >= 0)),
            "col2": Column(String, Check(lambda s: s >= 2)),
        },
        strict=True,
    )
    df_schema_columns_in_different_order = DataFrameSchema(
        {
            "col2": Column(String, Check(lambda s: s >= 2)),
            "col1": Column(Int, Check(lambda s: s >= 0)),
        },
        strict=True,
    )
    series_schema = SeriesSchema(
        String,
        checks=[Check(lambda s: s.str.startswith("foo"))],
        nullable=False,
        allow_duplicates=True,
        name="my_series",
    )
    series_schema_base = SeriesSchemaBase(
        String,
        checks=[Check(lambda s: s.str.startswith("foo"))],
        nullable=False,
        allow_duplicates=True,
        name="my_series",
    )
    not_equal_schema = DataFrameSchema({"col1": Column(String)}, strict=False)

    assert df_schema == copy.deepcopy(df_schema)
    assert df_schema != not_equal_schema
    assert df_schema == df_schema_columns_in_different_order
    assert series_schema == copy.deepcopy(series_schema)
    assert series_schema != not_equal_schema
    assert series_schema_base == copy.deepcopy(series_schema_base)
    assert series_schema_base != not_equal_schema


def test_add_and_remove_columns():
    """Check that adding and removing columns works as expected and doesn't
    modify the original underlying DataFrameSchema."""
    schema1 = DataFrameSchema(
        {
            "col1": Column(Int, Check(lambda s: s >= 0)),
        },
        strict=True,
    )

    schema1_exact_copy = copy.deepcopy(schema1)

    # test that add_columns doesn't modify schema1 after add_columns:
    schema2 = schema1.add_columns(
        {
            "col2": Column(String, Check(lambda x: x <= 0)),
            "col3": Column(Object, Check(lambda x: x == 0)),
        }
    )

    schema2_exact_copy = copy.deepcopy(schema2)

    assert schema1 == schema1_exact_copy

    # test that add_columns changed schema1 into schema2:
    expected_schema_2 = DataFrameSchema(
        {
            "col1": Column(Int, Check(lambda s: s >= 0)),
            "col2": Column(String, Check(lambda x: x <= 0)),
            "col3": Column(Object, Check(lambda x: x == 0)),
        },
        strict=True,
    )

    assert schema2 == expected_schema_2

    # test that remove_columns doesn't modify schema2:
    schema3 = schema2.remove_columns(["col2"])

    assert schema2 == schema2_exact_copy

    # test that remove_columns has removed the changes as expected:
    expected_schema_3 = DataFrameSchema(
        {
            "col1": Column(Int, Check(lambda s: s >= 0)),
            "col3": Column(Object, Check(lambda x: x == 0)),
        },
        strict=True,
    )

    assert schema3 == expected_schema_3

    # test that remove_columns can remove two columns:
    schema4 = schema2.remove_columns(["col2", "col3"])

    expected_schema_4 = DataFrameSchema(
        {"col1": Column(Int, Check(lambda s: s >= 0))}, strict=True
    )

    assert schema4 == expected_schema_4 == schema1


def test_schema_get_dtype():
    """Test that schema dtype and get_dtype methods handle regex columns."""
    schema = DataFrameSchema(
        {
            "col1": Column(Int),
            "var*": Column(Float, regex=True),
        }
    )

    data = pd.DataFrame(
        {
            "col1": [1, 2, 3],
            "var1": [1.0, 1.1, 1.2],
            "var2": [1.0, 1.1, 1.2],
            "var3": [1.0, 1.1, 1.2],
        }
    )

    with pytest.warns(UserWarning) as record:
        assert schema.dtype == {"col1": Int.str_alias}
    assert len(record) == 1
    assert (
        record[0]
        .message.args[0]
        .startswith("Schema has columns specified as regex column names:")
    )

    assert schema.get_dtype(data) == {
        "col1": Int.str_alias,
        "var1": Float.str_alias,
        "var2": Float.str_alias,
        "var3": Float.str_alias,
    }


def _boolean_update_column_case(bool_kwarg):
    def _assert_bool_case(old_schema, new_schema):
        assert not getattr(old_schema.columns["col"], bool_kwarg)
        assert getattr(new_schema.columns["col"], bool_kwarg)

    return [
        Column(Int, **{bool_kwarg: False}),
        "col",
        {bool_kwarg: True},
        _assert_bool_case,
    ]


@pytest.mark.parametrize(
    "column, column_to_update, update, assertion_fn",
    [
        [
            Column(Int),
            "col",
            {"pandas_dtype": String},
            lambda old, new: [
                old.columns["col"].pandas_dtype is Int,
                new.columns["col"].pandas_dtype is String,
            ],
        ],
        *[
            _boolean_update_column_case(bool_kwarg)
            for bool_kwarg in [
                "nullable",
                "allow_duplicates",
                "coerce",
                "required",
                "regex",
            ]
        ],
        [
            Column(Int, checks=Check.greater_than(0)),
            "col",
            {"checks": Check.less_than(10)},
            lambda old, new: [
                old.columns["col"].checks == [Check.greater_than(0)],
                new.columns["col"].checks == [Check.less_than(10)],
            ],
        ],
        # error cases
        [Column(Int), "col", {"name": "renamed_col"}, ValueError],
        [Column(Int), "foobar", {}, ValueError],
    ],
)
def test_dataframe_schema_update_column(
    column, column_to_update, update, assertion_fn
):
    """Test that DataFrameSchema columns create updated copies."""
    schema = DataFrameSchema({"col": column})
    if assertion_fn is ValueError:
        with pytest.raises(ValueError):
            schema.update_column(column_to_update, **update)
        return

    new_schema = schema.update_column(column_to_update, **update)
    assertion_fn(schema, new_schema)


def test_rename_columns():
    """Check that DataFrameSchema.rename_columns() method does its job"""

    rename_dict = {"col1": "col1_new_name", "col2": "col2_new_name"}
    schema_original = DataFrameSchema(
        columns={"col1": Column(Int), "col2": Column(Float)}
    )

    schema_renamed = schema_original.rename_columns(rename_dict)

    # Check if new column names are indeed present in the new schema
    assert all(
        [
            col_name in rename_dict.values()
            for col_name in schema_renamed.columns
        ]
    )
    # Check if original schema didn't change in the process
    assert all(
        [col_name in schema_original.columns for col_name in rename_dict]
    )

    with pytest.raises(errors.SchemaInitError):
        schema_original.rename_columns({"foo": "bar"})


@pytest.mark.parametrize(
    "select_columns, schema",
    [
        (
            ["col1", "col2"],
            DataFrameSchema(
                columns={
                    "col1": Column(Int),
                    "col2": Column(Int),
                    "col3": Column(Int),
                }
            ),
        ),
        (
            [("col1", "col1b"), ("col2", "col2b")],
            DataFrameSchema(
                columns={
                    ("col1", "col1a"): Column(Int),
                    ("col1", "col1b"): Column(Int),
                    ("col2", "col2a"): Column(Int),
                    ("col2", "col2b"): Column(Int),
                }
            ),
        ),
    ],
)
def test_select_columns(select_columns, schema):
    """Check that select_columns method correctly creates new subset schema."""
    original_columns = list(schema.columns)
    schema_selected = schema.select_columns(select_columns)
    assert all(x in select_columns for x in schema_selected.columns)
    assert all(x in original_columns for x in schema.columns)

    with pytest.raises(errors.SchemaInitError):
        schema.select_columns(["foo", "bar"])


def test_lazy_dataframe_validation_error():
    """Test exceptions on lazy dataframe validation."""
    schema = DataFrameSchema(
        columns={
            "int_col": Column(Int, Check.greater_than(5)),
            "int_col2": Column(Int),
            "float_col": Column(Float, Check.less_than(0)),
            "str_col": Column(String, Check.isin(["foo", "bar"])),
            "not_in_dataframe": Column(Int),
        },
        checks=Check(
            lambda df: df != 1, error="dataframe_not_equal_1", ignore_na=False
        ),
        index=Index(String, name="str_index"),
        strict=True,
    )

    dataframe = pd.DataFrame(
        data={
            "int_col": [1, 2, 6],
            "int_col2": ["a", "b", "c"],
            "float_col": [1.0, -2.0, 3.0],
            "str_col": ["foo", "b", "c"],
            "unknown_col": [None, None, None],
        },
        index=pd.Index(["index0", "index1", "index2"], name="str_index"),
    )

    expectation = {
        # schema object context -> check failure cases
        "DataFrameSchema": {
            # check name -> failure cases
            "column_in_schema": ["unknown_col"],
            "dataframe_not_equal_1": [1],
            "column_in_dataframe": ["not_in_dataframe"],
        },
        "Column": {
            "greater_than(5)": [1, 2],
            "pandas_dtype('int64')": ["object"],
            "less_than(0)": [1, 3],
        },
    }

    with pytest.raises(
        errors.SchemaErrors, match="^A total of .+ schema errors were found"
    ):
        schema.validate(dataframe, lazy=True)

    try:
        schema.validate(dataframe, lazy=True)
    except errors.SchemaErrors as err:

        # data in the caught exception should be equal to the dataframe
        # passed into validate
        assert err.data.equals(dataframe)

        # make sure all expected check errors are in schema errors
        for schema_context, check_failure_cases in expectation.items():
            err_df = err.schema_errors.loc[
                err.schema_errors.schema_context == schema_context
            ]
            for check, failure_cases in check_failure_cases.items():
                assert check in err_df.check.values
                assert (
                    err_df.loc[err_df.check == check]
                    .failure_case.isin(failure_cases)
                    .all()
                )


def test_lazy_dataframe_validation_nullable():
    """
    Test that non-nullable column failure cases are correctly processed during
    lazy validation.
    """
    schema = DataFrameSchema(
        columns={
            "int_column": Column(Int, nullable=False),
            "float_column": Column(Float, nullable=False),
            "str_column": Column(String, nullable=False),
        },
        strict=True,
    )

    df = pd.DataFrame(
        {
            "int_column": [1, None, 3],
            "float_column": [0.1, 1.2, None],
            "str_column": [None, "foo", "bar"],
        }
    )

    try:
        schema.validate(df, lazy=True)
    except errors.SchemaErrors as err:
        assert err.schema_errors.failure_case.isna().all()
        for col, index in [
            ("int_column", 1),
            ("float_column", 2),
            ("str_column", 0),
        ]:
            # pylint: disable=cell-var-from-loop
            assert (
                err.schema_errors.loc[
                    lambda df: df.column == col, "index"
                ].iloc[0]
                == index
            )


@pytest.mark.parametrize(
    "schema_cls, data",
    [
        [DataFrameSchema, pd.DataFrame({"column": [1]})],
        [SeriesSchema, pd.Series([1, 2, 3])],
        [partial(Column, name="column"), pd.DataFrame({"column": [1]})],
        [
            partial(Index, name="index"),
            pd.DataFrame(index=pd.Index([1, 2, 3], name="index")),
        ],
    ],
)
def test_lazy_dataframe_scalar_false_check(schema_cls, data):
    """Lazy validation handles checks returning scalar False values."""
    # define a check that always returns a scalare False value
    check = Check(
        check_fn=lambda _: False, element_wise=False, error="failing check"
    )
    schema = schema_cls(checks=check)
    with pytest.raises(errors.SchemaErrors):
        schema(data, lazy=True)


@pytest.mark.parametrize(
    "schema, data, expectation",
    [
        [
            SeriesSchema(Int, checks=Check.greater_than(0)),
            pd.Series(["a", "b", "c"]),
            {
                "data": pd.Series(["a", "b", "c"]),
                "schema_errors": {
                    # schema object context -> check failure cases
                    "SeriesSchema": {
                        # check name -> failure cases
                        "greater_than(0)": [
                            "TypeError(\"'>' not supported between instances of "
                            "'str' and 'int'\")",
                            # TypeError raised in python=3.5
                            'TypeError("unorderable types: str() > int()")',
                        ],
                        "pandas_dtype('int64')": ["object"],
                    },
                },
            },
        ],
        [
            Column(
                Int,
                checks=[Check.greater_than(1), Check.less_than(3)],
                name="column",
            ),
            pd.DataFrame({"column": [1, 2, 3]}),
            {
                "data": pd.DataFrame({"column": [1, 2, 3]}),
                "schema_errors": {
                    "Column": {"greater_than(1)": [1], "less_than(3)": [3]},
                },
            },
        ],
        [
            Index(String, checks=Check.isin(["a", "b", "c"])),
            pd.DataFrame({"col": [1, 2, 3]}, index=["a", "b", "d"]),
            {
                # expect that the data in the SchemaError is the pd.Index cast
                # into a Series
                "data": pd.Series(["a", "b", "d"]),
                "schema_errors": {
                    "Index": {f"isin({set(['a', 'b', 'c'])})": ["d"]},
                },
            },
        ],
        [
            MultiIndex(
                indexes=[
                    Index(Int, checks=Check.greater_than(0), name="index0"),
                    Index(Int, checks=Check.less_than(0), name="index1"),
                ]
            ),
            pd.DataFrame(
                {"column": [1, 2, 3]},
                index=pd.MultiIndex.from_arrays(
                    [[0, 1, 2], [-2, -1, 0]],
                    names=["index0", "index1"],
                ),
            ),
            {
                # expect that the data in the SchemaError is the pd.MultiIndex cast
                # into a DataFrame
                "data": pd.DataFrame(
                    {"column": [1, 2, 3]},
                    index=pd.MultiIndex.from_arrays(
                        [[0, 1, 2], [-2, -1, 0]],
                        names=["index0", "index1"],
                    ),
                ),
                "schema_errors": {
                    "MultiIndex": {
                        "greater_than(0)": [0],
                        "less_than(0)": [0],
                    },
                },
            },
        ],
    ],
)
def test_lazy_series_validation_error(schema, data, expectation):
    """Test exceptions on lazy series validation."""
    try:
        schema.validate(data, lazy=True)
    except errors.SchemaErrors as err:
        # data in the caught exception should be equal to the data
        # passed into validate
        assert err.data.equals(expectation["data"])

        # make sure all expected check errors are in schema errors
        for schema_context, check_failure_cases in expectation[
            "schema_errors"
        ].items():
            assert schema_context in err.schema_errors.schema_context.values
            err_df = err.schema_errors.loc[
                err.schema_errors.schema_context == schema_context
            ]
            for check, failure_cases in check_failure_cases.items():
                assert check in err_df.check.values
                assert (
                    err_df.loc[err_df.check == check]
                    .failure_case.isin(failure_cases)
                    .all()
                )


def test_schema_transformer_deprecated():
    """Using the transformer argument should raise a deprecation warning."""
    with pytest.warns(DeprecationWarning):
        DataFrameSchema(transformer=lambda df: df)


@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize(
    "from_dtype,to_dtype",
    [
        [float, int],
        [int, float],
        [object, int],
        [object, float],
        [int, object],
        [float, object],
    ],
)
def test_schema_coerce_inplace_validation(inplace, from_dtype, to_dtype):
    """Test coercion logic for validation when inplace is True and False"""

    to_dtype = PandasDtype.from_python_type(to_dtype).str_alias
    from_dtype = PandasDtype.from_python_type(from_dtype).str_alias

    df = pd.DataFrame({"column": pd.Series([1, 2, 6], dtype=from_dtype)})
    schema = DataFrameSchema({"column": Column(to_dtype, coerce=True)})
    validated_df = schema.validate(df, inplace=inplace)

    assert validated_df["column"].dtype == to_dtype
    if inplace:
        # inplace mutates original dataframe
        assert df["column"].dtype == to_dtype
    else:
        # not inplace preserves original dataframe type
        assert df["column"].dtype == from_dtype


@pytest.fixture
def schema_simple():
    """Simple schema fixture."""
    schema = DataFrameSchema(
        columns={
            "col1": Column(pandas_dtype=Int),
            "col2": Column(pandas_dtype=Float),
        },
        index=Index(pandas_dtype=String, name="ind0"),
    )
    return schema


@pytest.fixture
def schema_multiindex():
    """Fixture for schema with MultiIndex."""
    schema = DataFrameSchema(
        columns={
            "col1": Column(pandas_dtype=Int),
            "col2": Column(pandas_dtype=Float),
        },
        index=MultiIndex(
            [
                Index(pandas_dtype=String, name="ind0"),
                Index(pandas_dtype=String, name="ind1"),
            ]
        ),
    )
    return schema


@pytest.mark.parametrize("drop", [True, False])
def test_set_index_drop(drop, schema_simple):
    """Test that setting index correctly handles column dropping."""
    test_schema = schema_simple.set_index(keys=["col1"], drop=drop)
    if drop is True:
        assert len(test_schema.columns) == 1
        assert list(test_schema.columns.keys()) == ["col2"]
    else:
        assert len(test_schema.columns) == 2
        assert list(test_schema.columns.keys()) == ["col1", "col2"]
        assert test_schema.index.name == "col1"


@pytest.mark.parametrize("append", [True, False])
def test_set_index_append(append, schema_simple):
    """
    Test that setting index correctly handles appending to existing index.
    """
    test_schema = schema_simple.set_index(keys=["col1"], append=append)
    if append is True:
        assert isinstance(test_schema.index, MultiIndex)
        assert list(test_schema.index.columns.keys()) == ["ind0", "col1"]
        assert (
            test_schema.index.columns["col1"].dtype
            == schema_simple.columns["col1"].dtype
        )
    else:
        assert isinstance(test_schema.index, Index)
        assert test_schema.index.name == "col1"


@pytest.mark.parametrize("drop", [True, False])
def test_reset_index_drop(drop, schema_simple):
    """Test that resetting index correctly handles dropping index levels."""
    test_schema = schema_simple.reset_index(drop=drop)
    if drop:
        assert len(test_schema.columns) == 2
        assert list(test_schema.columns.keys()) == ["col1", "col2"]
    else:
        assert len(test_schema.columns) == 3
        assert list(test_schema.columns.keys()) == ["col1", "col2", "ind0"]
        assert test_schema.index is None


def test_reset_index_level(schema_multiindex):
    """Test that resetting index correctly handles specifying level."""
    test_schema = schema_multiindex.reset_index(level=["ind0"])
    assert test_schema.index.name == "ind1"
    assert isinstance(test_schema.index, Index)

    test_schema = schema_multiindex.reset_index(level=["ind0", "ind1"])
    assert test_schema.index is None
    assert set(list(test_schema.columns.keys())) == set(
        ["col1", "col2", "ind0", "ind1"]
    )


def test_invalid_keys(schema_simple):
    """Test that re/set_index raises expected exceptions."""
    with pytest.raises(errors.SchemaInitError):
        schema_simple.set_index(["foo", "bar"])
    with pytest.raises(TypeError):
        schema_simple.set_index()
    with pytest.raises(errors.SchemaInitError):
        schema_simple.reset_index(["foo", "bar"])

    schema_simple.index = None
    with pytest.raises(errors.SchemaInitError):
        schema_simple.reset_index()


def test_update_columns(schema_simple):
    """ Catch-all test for update columns functionality """

    # Basic function
    test_schema = schema_simple.update_columns({"col2": {"pandas_dtype": Int}})
    assert (
        schema_simple.columns["col1"].properties
        == test_schema.columns["col1"].properties
    )
    assert test_schema.columns["col2"].pandas_dtype == Int

    # Multiple columns, multiple properties
    test_schema = schema_simple.update_columns(
        {
            "col1": {"pandas_dtype": Category, "coerce": True},
            "col2": {"pandas_dtype": Int, "allow_duplicates": False},
        }
    )
    assert test_schema.columns["col1"].pandas_dtype == Category
    assert test_schema.columns["col1"].coerce is True
    assert test_schema.columns["col2"].pandas_dtype == Int
    assert test_schema.columns["col2"].allow_duplicates is False

    # Errors
    with pytest.raises(errors.SchemaInitError):
        schema_simple.update_columns({"col3": {"pandas_dtype": Int}})
    with pytest.raises(errors.SchemaInitError):
        schema_simple.update_columns({"col1": {"name": "foo"}})
    with pytest.raises(errors.SchemaInitError):
        schema_simple.update_columns({"ind0": {"pandas_dtype": Int}})


@pytest.mark.parametrize("pdtype", list(PandasDtype) + [None])  # type: ignore
def test_series_schema_pdtype(pdtype):
    """Series schema pdtype property should return PandasDtype."""
    if pdtype is None:
        series_schema = SeriesSchema(pdtype)
        assert series_schema.pdtype is None
        return
    for pandas_dtype_input in [
        pdtype,
        pdtype.str_alias,
        pdtype.value,
    ]:
        series_schema = SeriesSchema(pandas_dtype_input)
        if pdtype is PandasDtype.STRING and LEGACY_PANDAS:
            assert series_schema.pdtype == PandasDtype.String
        else:
            assert series_schema.pdtype == pdtype


@pytest.mark.parametrize(
    "data, error",
    [
        [
            pd.DataFrame(
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=["a", "a", "b"]
            ),
            None,
        ],
        [
            pd.DataFrame(
                [[1, 2, 3], list("xyz"), [7, 8, 9]], columns=["a", "a", "b"]
            ),
            errors.SchemaError,
        ],
    ],
)
@pytest.mark.parametrize(
    "schema",
    [
        DataFrameSchema({"a": Column(int), "b": Column(int)}),
        DataFrameSchema({"a": Column(int, coerce=True), "b": Column(int)}),
        DataFrameSchema({"a": Column(int, regex=True), "b": Column(int)}),
    ],
)
def test_dataframe_duplicated_columns(data, error, schema):
    """Test that schema can handle dataframes with duplicated columns."""
    if error is None:
        assert isinstance(schema(data), pd.DataFrame)
    else:
        with pytest.raises(error):
            schema(data)
