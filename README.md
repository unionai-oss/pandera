pandera
=======

Validating pandas data structures for people seeking correct things.

A light-weight and flexible validation package for
<a href="http://pandas.pydata.org" target="_blank">pandas</a> data structures,
built on top of
<a href="https://github.com/keleshev/schema" target="_blank">schema</a>,
a powerful Python data structure validation tool. API inspired by
<a href="https://tmiguelt.github.io/PandasSchema" target="_blank">pandas-schema</a>


Why?
----

Because pandas data structures hide a lot of information, and explicitly
validating them in production-critical or reproducible research settings is
a good idea.

And it also makes it easier to review pandas code :)


Install
-------

```
pip install pandera
```


Example Usage
-------------

### `pandera.DataFrameSchema`

Basic Usage: validate a `pandas.DataFrame`

```python
import pandas as pd

from pandera import Column, DataFrameSchema, PandasDtype


# validate columns
schema = DataFrameSchema(
    columns=[
        Column("column1", PandasDtype.Int, lambda x: 0 <= x <= 10),
        Column("column2", PandasDtype.Float, lambda x: x < -1.2),
        Column("column3", PandasDtype.String, lambda x: x.startswith("value_"))
    ],
)

df = pd.DataFrame({
    "column1": [1, 4, 0, 10, 9],
    "column2": [-1.3, -1.4, -2.9, -10.1, -20.4],
    "column3": ["value_1", "value_2", "value_3", "value_2", "value_1"]
})

validated_df = schema.validate(df)
print(validated_df)

#  Output:
#     column1  column2  column3
#  0        1     -1.3  value_1
#  1        4     -1.4  value_2
#  2        0     -2.9  value_3
#  3       10    -10.1  value_2
#  4        9    -20.4  value_1
```

### Errors

If the dataframe does not pass validation checks, `pandera` provides useful
error messages.

```python
simple_schema = DataFrameSchema(
    columns=[Column("column1", PandasDtype.Int, lambda x: 0 <= x <= 10)])


# validation rule violated
fail_check_df = pd.DataFrame({
    "column1": [-20, 5, 10, 30],
})

simple_schema.validate(fail_check_df)

#  Output:
#
#  SchemaError: series did not pass element-wise validator
#  'columns=[Column("column1", PandasDtype.Int, lambda x: 0 <= x <= 10)])'.
#  failure cases: {0: -20, 3: 30}


# column name mis-specified
wrong_column_df = pd.DataFrame({
    "foo": ["bar"] * 10,
    "baz": [1] * 10
})

simple_schema.validate(wrong_column_df)

#  Output:
#
#  SchemaError: column 'column1' not in dataframe
#     foo  baz
#  0  bar    1
#  1  bar    1
#  2  bar    1
#  3  bar    1
#  4  bar    1
```


### `pandera.validate_input`

Decorator that validates input pandas DataFrame/Series before entering the
wrapped function.

```python
from pandera import validate_input


# use element_wise=False to apply a validator function that has access to the
# pd.Series API.
in_schema = DataFrameSchema(
    columns=[
        Column("column1", PandasDtype.Int, lambda x: 0 <= x <= 10),
        Column("column2", PandasDtype.Float, lambda x: x < -1.2),
    ])


# provide the argument name as a string, or integer representing index
# in the positional arguments.
@validate_input(in_schema, "dataframe")
def preprocessor(dataframe):
    dataframe["column4"] = dataframe["column1"] + dataframe["column2"]
    return dataframe


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


### `pandera.validate_output`

The same as `validate_input`, but this decorator checks the output
DataFrame/Series of the decorated function.

```python
from pandera import validate_output


# assert that all elements in "column1" are zero
out_schema = DataFrameSchema(
    columns=[Column("column1", PandasDtype.Int, lambda x: x == 0)])


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
