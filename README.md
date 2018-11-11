# pandera

Validating pandas data structures for people seeking correct things.

A light-weight and flexible validation package for
<a href="http://pandas.pydata.org" target="_blank">pandas</a> data structures,
built on top of
<a href="https://github.com/keleshev/schema" target="_blank">schema</a>,
a powerful Python data structure validation tool. API inspired by
<a href="https://tmiguelt.github.io/PandasSchema" target="_blank">pandas-schema</a>.


## Why?

Because pandas data structures hide a lot of information, and explicitly
validating them in production-critical or reproducible research settings is
a good idea.

And it also makes it easier to review pandas code :)


## Install

```
pip install pandera
```


## Example Usage

### `DataFrameSchema`

```python
import pandas as pd

from pandera import Column, DataFrameSchema, PandasDtype, Validator


# validate columns
schema = DataFrameSchema([
    Column("column1", PandasDtype.Int, Validator(lambda x: 0 <= x <= 10)),
    Column("column2", PandasDtype.Float, Validator(lambda x: x < -1.2)),
    Column("column3", PandasDtype.String,
           Validator(lambda x: x.startswith("value_")))
])

# optionally, you can pass strings representing the legal pandas datatypes:
# http://pandas.pydata.org/pandas-docs/stable/basics.html#dtypes
schema = DataFrameSchema([
    Column("column1", "int[64]", Validator(lambda x: 0 <= x <= 10)),
    ...
])

df = pd.DataFrame({
    "column1": [1, 4, 0, 10, 9],
    "column2": [-1.3, -1.4, -2.9, -10.1, -20.4],
    "column3": ["value_1", "value_2", "value_3", "value_2", "value_1"]
})

validated_df = schema.validate(df)
print(validated_df)

#     column1  column2  column3
#  0        1     -1.3  value_1
#  1        4     -1.4  value_2
#  2        0     -2.9  value_3
#  3       10    -10.1  value_2
#  4        9    -20.4  value_1
```

#### Validating DataFrame Index

You can also specify an `Index` in the `DataFrameSchema`.

```python
from pandera import Index

schema = DataFrameSchema(
    columns=[Column("a", PandasDtype.Int)],
    index=Index(
        PandasDtype.String,
        Validator(lambda x: x.startswith("index_"))))

df = pd.DataFrame({"a": [1, 2, 3]}, index=["index_1", "index_2", "index_3"])

print(schema.validate(df))

#          a
# index_1  1
# index_2  2
# index_3  3


df.index = ["foo1", "foo2", "foo3"]
schema.validate(df)

# SchemaError: <Schema Index> failed element-wise validator 0:
# <lambda>
# failure cases: {0: 'foo1', 1: 'foo2', 2: 'foo3'}
```


#### Informative Errors

If the dataframe does not pass validation checks, `pandera` provides useful
error messages. An `error` argument can also be supplied to `Validator` for
custom error messages.

```python
simple_schema = DataFrameSchema([
    Column("column1", PandasDtype.Int,
           Validator(lambda x: 0 <= x <= 10, error="range checker [0, 10]"))
])

# validation rule violated
fail_check_df = pd.DataFrame({
    "column1": [-20, 5, 10, 30],
})

simple_schema.validate(fail_check_df)

# schema.SchemaError: series failed element-wise validator 0:
# <lambda>: range checker [0, 10]
# failure cases: {0: -20, 3: 30}


# column name mis-specified
wrong_column_df = pd.DataFrame({
    "foo": ["bar"] * 10,
    "baz": [1] * 10
})

simple_schema.validate(wrong_column_df)

#  SchemaError: column 'column1' not in dataframe
#     foo  baz
#  0  bar    1
#  1  bar    1
#  2  bar    1
#  3  bar    1
#  4  bar    1
```

### Nullable Columns

By default, SeriesSchema/Column objects assume that values are not nullable.
In order to accept null values, you need to explicitly specify `nullable=True`,
or else you'll get an error.

```python
df = pd.DataFrame({"column1": [5, 1, np.nan]})

non_null_schema = DataFrameSchema([
    Column("column1", PandasDtype.Int, Validator(lambda x: x > 0))])

non_null_schema.validate(df)

# SchemaError: non-nullable series contains null values: {2: nan}
```

**NOTE:** Due to a known limitation in
[pandas](http://pandas.pydata.org/pandas-docs/stable/gotchas.html#support-for-integer-na),
integer arrays cannot contain `NaN` values, so this schema will return a
dataframe where `column1` is of type `float`.

```python
null_schema = DataFrameSchema([
    Column("column1", PandasDtype.Int, Validator(lambda x: x > 0),
           nullable=True)])

null_schema.validate(df)

#    column1
# 0      5.0
# 1      1.0
# 2      NaN
```


### `SeriesSchema`

```python
from pandera import SeriesSchema

# specify multiple validators
schema = SeriesSchema(
    PandasDtype.String, [
        Validator(lambda x: "foo" in x),
        Validator(lambda x: x.endswith("bar")),
        Validator(lambda x: len(x) > 3)])

schema.validate(pd.Series(["1_foobar", "2_foobar", "3_foobar"]))

#  0    1_foobar
#  1    2_foobar
#  2    3_foobar
#  dtype: object
```


### Vectorized Validators

If you need to make basic statistical assertions about a column, or you want
to take advantage of the speed gains affarded by the pd.Series API, use the
`element_wise=False` keyword argument. The signature of validators then becomes
`pd.Series -> bool|pd.Series[bool]`.

```python
schema = DataFrameSchema([
    Column("a", PandasDtype.Int,
           [
                # this validator returns a bool
                Validator(lambda s: s.mean() > 5, element_wise=False),
                # this validator returns a boolean series
                Validator(lambda s: s > 0, element_wise=False)])
])

df = pd.DataFrame({"a": [4, 4, 5, 6, 6, 7, 8, 9]})
schema.validate(df)
```


## Plugging into Existing Workflows

If you have an existing data pipeline that uses pandas data structures, you can
use the `validate_input` and `validate_output` decorators to easily check
function arguments or returned variables from existing functions.


### `validate_input`

Validates input pandas DataFrame/Series before entering the wrapped function.

```python
from pandera import validate_input


df = pd.DataFrame({
    "column1": [1, 4, 0, 10, 9],
    "column2": [-1.3, -1.4, -2.9, -10.1, -20.4],
})

in_schema = DataFrameSchema([
    Column("column1", PandasDtype.Int, Validate(lambda x: 0 <= x <= 10)),
    Column("column2", PandasDtype.Float, Validate(lambda x: x < -1.2)),
])

# by default, assumes that the first argument is dataframe/series.
@validate_input(in_schema)
def preprocessor(dataframe):
    dataframe["column4"] = dataframe["column1"] + dataframe["column2"]
    return dataframe

# or you can provide the argument name as a string
@validate_input(in_schema, "dataframe")
def preprocessor(dataframe):
    ...

# or integer representing index in the positional arguments.
@validate_input(in_schema, 1)
def preprocessor(foo, dataframe):
    ...

preprocessed_df = preprocessor(df)
print(preprocessed_df)

#  Output:
#     column1  column2  column3  column4
#  0        1     -1.3  value_1     -0.3
#  1        4     -1.4  value_2      2.6
#  2        0     -2.9  value_3     -2.9
#  3       10    -10.1  value_2     -0.1
#  4        9    -20.4  value_1    -11.4
```


### `validate_output`

The same as `validate_input`, but this decorator checks the output
DataFrame/Series of the decorated function.

```python
from pandera import validate_output


# assert that all elements in "column1" are zero
out_schema = DataFrameSchema([
    Column("column1", PandasDtype.Int, Validate(lambda x: x == 0))])


# by default assumes that the pandas DataFrame/Schema is the only output
@validate_output(out_schema)
def zero_column_1(df):
    df["column1"] = 0
    return df


# you can also specify in the index of the argument if the output is list-like
@validate_output(out_schema, 1)
def zero_column_1_arg(df):
    df["column1"] = 0
    return "foobar", df


# or the key containing the data structure to verify if the output is dict-like
@validate_output(out_schema, "out_df")
def zero_column_1_dict(df):
    df["column1"] = 0
    return {"out_df": df, "out_str": "foobar"}


# for more complex outputs, you can specify a function
@validate_output(out_schema, lambda x: x[1]["out_df"])
def zero_column_1_custom(df):
    df["column1"] = 0
    return ("foobar", {"out_df": df})


zero_column_1(preprocessed_df)
zero_column_1_arg(preprocessed_df)
zero_column_1_dict(preprocessed_df)
zero_column_1_custom(preprocessed_df)
```

Tests
-----

```
pip install pytest
pytest tests
```

Issues
------

Go <a href="https://github.com/cosmicBboy/pandera/issues" target="_blank">here</a>
to submit feature requests or bugfixes.
